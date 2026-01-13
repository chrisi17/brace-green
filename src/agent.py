import os
from typing import Any
from pydantic import BaseModel, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from .messenger import Messenger
from .evaluator.workflow import EvaluatorWorkflow
from .evaluator.agent_interface import create_agent_interface
from .evaluator.step_evaluator import StepEvaluator
from .evaluator.utils import discover_all_challenges


class EvalRequest(BaseModel):
    """Request format for BraceGreen evaluator."""
    challenges: list[str]  # List of challenge names or ["all"]
    agent_config: dict[str, Any]  # Agent configuration
    evaluator_config: dict[str, Any] = {}  # Optional evaluator config
    max_iterations: int = 10  # Max iterations per step
    enable_phoenix: bool = False  # Whether to enable Phoenix tracing (requires Phoenix server)


class Agent:
    """BraceGreen evaluator agent."""
    
    required_roles: list[str] = []
    required_config_keys: list[str] = []

    def __init__(self, writeups_path: str = "./data/agentbeats"):
        self.messenger = Messenger()
        self.writeups_path = writeups_path

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the incoming evaluation request."""
        if not request.challenges:
            return False, "No challenges specified."

        if request.challenges == ["all"] or request.challenges == "all":
            # If "all" is specified, discover challenges
            discovered_challenges = discover_all_challenges(self.writeups_path)
            if not discovered_challenges:
                return False, f"No challenges found in {self.writeups_path} to evaluate 'all'."
        else:
            # Validate specific challenges exist
            all_available_challenges = discover_all_challenges(self.writeups_path)
            missing_challenges = [c for c in request.challenges if c not in all_available_challenges]
            if missing_challenges:
                return False, f"Missing challenges: {', '.join(missing_challenges)}"

        if "mode" not in request.agent_config:
            return False, "Agent configuration missing 'mode' (e.g., 'internal' or 'a2a')."

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Execute the evaluation logic based on the incoming message."""
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        await updater.update_status(
            TaskState.working, new_agent_text_message("Starting evaluation setup...")
        )

        # Resolve challenge list
        challenges_to_evaluate = request.challenges
        if challenges_to_evaluate == ["all"] or challenges_to_evaluate == "all":
            challenges_to_evaluate = discover_all_challenges(self.writeups_path)
            await updater.update_status(
                TaskState.working, new_agent_text_message(f"Evaluating all {len(challenges_to_evaluate)} challenges.")
            )
        else:
            await updater.update_status(
                TaskState.working, new_agent_text_message(f"Evaluating {len(challenges_to_evaluate)} challenge(s): {', '.join(challenges_to_evaluate)}")
            )

        # Create agent interface
        print(f"Creating agent interface in {request.agent_config['mode']} mode...")
        agent_interface = create_agent_interface(request.agent_config)

        # Create step evaluator
        print("Creating step evaluator...")
        evaluator_model = request.evaluator_config.get("model", "gpt-4o")
        evaluator_max_tokens = request.evaluator_config.get("max_tokens", 2000)
        api_key = request.evaluator_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = request.evaluator_config.get("base_url") or os.getenv("OPENAI_BASE_URL")

        step_evaluator = StepEvaluator(
            model=evaluator_model,
            max_tokens=evaluator_max_tokens,
            api_key=api_key,
            base_url=base_url
        )

        # Create workflow
        print("Building evaluation workflow...")
        enable_phoenix = request.enable_phoenix
        workflow = EvaluatorWorkflow(
            agent_interface=agent_interface,
            step_evaluator=step_evaluator,
            max_iterations_per_step=request.max_iterations,
            enable_phoenix=enable_phoenix
        )

        all_results = []
        for challenge_name in challenges_to_evaluate:
            await updater.update_status(
                TaskState.working, new_agent_text_message(f"Starting challenge: {challenge_name}")
            )

            try:
                agent_llm_config = {
                    "model": request.agent_config.get("model", "gpt-4o"),
                    "temperature": request.agent_config.get("temperature", 0.7),
                    "max_tokens": request.agent_config.get("max_tokens", 500),
                }

                evaluator_llm_config = {
                    "model": evaluator_model,
                    "max_tokens": evaluator_max_tokens
                }

                # Run workflow in thread to avoid blocking the event loop
                import asyncio
                print(f"Starting evaluation for {challenge_name}...")
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,  # Use default ThreadPoolExecutor
                    workflow.run,
                    challenge_name,
                    agent_llm_config,
                    evaluator_llm_config
                )

                all_results.append(result)
                print(f"✓ Completed {challenge_name}: Score = {result['score']:.2%}")
                
                await updater.update_status(
                    TaskState.working, new_agent_text_message(f"Completed {challenge_name}: Score = {result['score']:.2%}")
                )

            except Exception as e:
                error_result = {
                    "challenge": challenge_name,
                    "error": str(e),
                    "score": 0.0
                }
                all_results.append(error_result)
                print(f"✗ Failed {challenge_name}: {e}")
                
                await updater.update_status(
                    TaskState.working, new_agent_text_message(f"Failed {challenge_name}: {e}")
                )

        # Aggregate results
        total_score = sum(r.get("score", 0.0) for r in all_results) / len(all_results) if all_results else 0.0

        # Create summary text
        summary_lines = [
            f"Evaluation completed for {len(challenges_to_evaluate)} challenge(s)",
            f"Overall Score: {total_score:.2%}",
            "",
            "Results:"
        ]
        for result in all_results:
            challenge = result.get("challenge", "unknown")
            score = result.get("score", 0.0)
            if "error" in result:
                summary_lines.append(f"  - {challenge}: FAILED ({result['error']})")
            else:
                summary_lines.append(f"  - {challenge}: {score:.2%}")

        summary_text = "\n".join(summary_lines)

        # Add final artifact with results
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data={
                    "overall_score": total_score,
                    "challenges_evaluated": len(challenges_to_evaluate),
                    "results": all_results
                }))
            ],
            name="Evaluation Result"
        )
