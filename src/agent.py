import os
from typing import Any
from pydantic import BaseModel, ValidationError, HttpUrl
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from .messenger import Messenger
from .evaluator.workflow import EvaluatorWorkflow
from .evaluator.agent_interface import create_agent_interface
from .evaluator.step_evaluator import StepEvaluator
from .evaluator.utils import discover_all_challenges


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class Agent:
    """BraceGreen evaluator agent."""
    
    # No required participant roles (we accept any agents to evaluate)
    required_roles: list[str] = []
    # Required config keys for evaluation
    required_config_keys: list[str] = ["challenges", "agent_config"]

    def __init__(self, writeups_path: str = "./data"):
        self.messenger = Messenger()
        self.writeups_path = writeups_path

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the incoming evaluation request."""
        # Standard validation from template
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Additional validation specific to our evaluator
        challenges = request.config.get("challenges", [])
        agent_config = request.config.get("agent_config", {})
        
        if not challenges:
            return False, "No challenges specified in config."

        if challenges == ["all"] or challenges == "all":
            # If "all" is specified, discover challenges
            discovered_challenges = discover_all_challenges(self.writeups_path)
            if not discovered_challenges:
                return False, f"No challenges found in {self.writeups_path} to evaluate 'all'."
        else:
            # Validate specific challenges exist
            all_available_challenges = discover_all_challenges(self.writeups_path)
            missing_challenges = [c for c in challenges if c not in all_available_challenges]
            if missing_challenges:
                return False, f"Missing challenges: {', '.join(missing_challenges)}"

        if "mode" not in agent_config:
            return False, "Agent configuration missing 'mode' (e.g., 'internal' or 'a2a')."

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
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

        # Extract config fields
        challenges = request.config.get("challenges", [])
        agent_config = request.config.get("agent_config", {})
        evaluator_config = request.config.get("evaluator_config", {})
        max_iterations = request.config.get("max_iterations", 10)
        enable_phoenix = request.config.get("enable_phoenix", False)
        include_goal = request.config.get("include_goal", "first")
        include_tactic = request.config.get("include_tactic", "first")
        include_prerequisites = request.config.get("include_prerequisites", "always")
        evaluation_protocol = request.config.get("evaluation_protocol", "match_alternatives")
        task_mode = request.config.get("task_mode", "command")
        history_context = request.config.get("history_context", ["goal", "command", "output", "results"])
        
        # Validate incompatible option combinations
        if task_mode == "goal" and include_goal == "always":
            error_msg = (
                "Invalid configuration: When task_mode is 'goal', include_goal cannot be 'always'. "
                "Otherwise the agent would always be given the answer it should predict. "
                "Use 'never' for full challenge, or 'first' to show one example then test."
            )
            await updater.reject(new_agent_text_message(error_msg))
            return
        
        # Production mode override (when running in Docker)
        is_production = os.getenv("BRACEGREEN_PRODUCTION", "false").lower() == "true"
        if is_production:
            # Enforce a2a mode in production
            agent_config["mode"] = "a2a"
            include_goal = "first"
            # Prohibit evaluator config population in production
            evaluator_config = {}
            print("Production mode: Enforcing a2a mode and using default evaluator config")
            await updater.update_status(
                TaskState.working, new_agent_text_message(f"Production mode: Enforcing a2a mode and using default evaluator config")
            )

        # Extract agent information from participants if in a2a mode
        if agent_config.get("mode") == "a2a" and request.participants:
            # participants is a dict mapping role -> URL
            # For now, we'll take the first participant as the agent to evaluate
            for role, url in request.participants.items():
                agent_config["agent_url"] = str(url)  # Convert HttpUrl to string
                agent_config["role"] = role
                break  # Use first participant as the agent

        # Resolve challenge list
        challenges_to_evaluate = challenges
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
        print(f"Creating agent interface in {agent_config['mode']} mode...")
        # Pass evaluation_protocol and task_mode to agent interface
        agent_config["evaluation_protocol"] = evaluation_protocol
        agent_config["task_mode"] = task_mode
        agent_interface = create_agent_interface(agent_config)

        # Create step evaluator
        print("Creating step evaluator...")
        evaluator_model = evaluator_config.get("model", "gpt-4o")
        evaluator_max_tokens = evaluator_config.get("max_tokens", 2000)
        api_key = evaluator_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = evaluator_config.get("base_url") or os.getenv("OPENAI_BASE_URL")

        step_evaluator = StepEvaluator(
            model=evaluator_model,
            max_tokens=evaluator_max_tokens,
            api_key=api_key,
            base_url=base_url,
            evaluation_protocol=evaluation_protocol,
            task_mode=task_mode
        )

        # Create workflow
        print("Building evaluation workflow...")
        workflow = EvaluatorWorkflow(
            agent_interface=agent_interface,
            step_evaluator=step_evaluator,
            max_iterations_per_step=max_iterations,
            enable_phoenix=enable_phoenix,
            include_goal=include_goal,
            include_tactic=include_tactic,
            include_prerequisites=include_prerequisites,
            history_context=history_context
        )

        all_results = []
        for challenge_name in challenges_to_evaluate:
            await updater.update_status(
                TaskState.working, new_agent_text_message(f"Starting challenge: {challenge_name}")
            )

            try:
                agent_llm_config = {
                    "model": agent_config.get("model", "gpt-4o"),
                    "temperature": agent_config.get("temperature", 0.0),
                    "max_tokens": agent_config.get("max_tokens", 500),
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
                    "max_iterations": max_iterations,
                    "include_goal": include_goal,
                    "include_tactic": include_tactic,
                    "include_prerequisites": include_prerequisites,
                    "history_context": history_context,
                    "evaluation_protocol": evaluation_protocol,
                    "task_mode": task_mode,
                    "timeout": agent_config.get("timeout"),
                    "results": all_results
                }))
            ],
            name="Evaluation Result"
        )
