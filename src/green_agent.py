"""Green agent A2A server for evaluating red-teaming agents.

This module provides an A2A-compatible server that wraps the evaluator workflow,
allowing external systems (like agentbeats) to request evaluations via the A2A protocol.
"""

import uvicorn
import tomllib
import json
import os
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# A2A SDK imports (based on tau-bench example pattern)
try:
    # Try using a simple HTTP API approach compatible with A2A protocol
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    from starlette.requests import Request
    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    print("Warning: Starlette not available. Install with: pip install starlette uvicorn")

from .evaluator.workflow import EvaluatorWorkflow
from .evaluator.agent_interface import create_agent_interface
from .evaluator.step_evaluator import StepEvaluator
from .evaluator.utils import discover_all_challenges

load_dotenv()


class GreenAgentExecutor:
    """Executor for green agent evaluation requests."""
    
    def __init__(self, writeups_path: str = "./data/agentbeats"):
        """Initialize the green agent executor.
        
        Args:
            writeups_path: Path to writeups directory
        """
        self.writeups_path = writeups_path
    
    async def execute(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an evaluation request.
        
        Args:
            request_data: Request containing:
                - challenges: List of challenge names or ["all"]
                - agent_config: Dict with agent configuration
                    - mode: "internal" or "a2a"
                    - For internal: model, temperature, max_tokens, etc.
                    - For a2a: agent_url, timeout
                - evaluator_config: Dict with evaluator LLM configuration (optional)
                - max_iterations: Max iterations per step (optional)
                
        Returns:
            Dictionary with evaluation results
        """
        # Parse request
        challenges_spec = request_data.get("challenges", [])
        agent_config = request_data.get("agent_config", {})
        evaluator_config = request_data.get("evaluator_config", {})
        max_iterations = request_data.get("max_iterations", 10)
        
        # Resolve challenge list
        if challenges_spec == ["all"] or challenges_spec == "all":
            challenges = discover_all_challenges(self.writeups_path)
            print(f"Evaluating all {len(challenges)} challenges")
        else:
            challenges = challenges_spec
            print(f"Evaluating {len(challenges)} challenge(s): {', '.join(challenges)}")
        
        # Create agent interface
        print(f"Creating agent interface in {agent_config.get('mode', 'internal')} mode...")
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
            base_url=base_url
        )
        
        # Create workflow
        print("Building evaluation workflow...")
        enable_phoenix = request_data.get("enable_phoenix", True)
        workflow = EvaluatorWorkflow(
            agent_interface=agent_interface,
            step_evaluator=step_evaluator,
            max_iterations_per_step=max_iterations,
            enable_phoenix=enable_phoenix
        )
        
        # Run evaluations
        all_results = []
        for challenge_name in challenges:
            print(f"\n{'=' * 70}")
            print(f"Evaluating challenge: {challenge_name}")
            print('=' * 70)
            
            try:
                # Build configs for workflow.run()
                agent_llm_config = {
                    "model": agent_config.get("model", "gpt-4o"),
                    "temperature": agent_config.get("temperature", 0.7),
                    "max_tokens": agent_config.get("max_tokens", 500),
                }
                
                evaluator_llm_config = {
                    "model": evaluator_model,
                    "max_tokens": evaluator_max_tokens
                }
                
                result = workflow.run(
                    challenge_name=challenge_name,
                    agent_llm_config=agent_llm_config,
                    evaluator_llm_config=evaluator_llm_config
                )
                
                all_results.append(result)
                print(f"✓ Completed {challenge_name}: Score = {result['score']:.2%}")
                
            except Exception as e:
                error_result = {
                    "challenge": challenge_name,
                    "error": str(e),
                    "score": 0.0
                }
                all_results.append(error_result)
                print(f"✗ Failed {challenge_name}: {e}")
        
        # Aggregate results
        total_score = sum(r.get("score", 0.0) for r in all_results) / len(all_results) if all_results else 0.0
        
        print(f"\n{'=' * 70}")
        print(f"Evaluation Complete")
        print(f"Overall Score: {total_score:.2%}")
        print('=' * 70)
        
        return {
            "status": "completed",
            "overall_score": total_score,
            "challenges_evaluated": len(challenges),
            "results": all_results
        }


def load_agent_card_toml(toml_path: str) -> Dict[str, Any]:
    """Load agent card from TOML file.
    
    Args:
        toml_path: Path to TOML file
        
    Returns:
        Agent card dictionary
    """
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


async def handle_agent_card(request: Request) -> JSONResponse:
    """Handle agent card request.
    
    Args:
        request: HTTP request
        
    Returns:
        JSON response with agent card
    """
    # Load agent card
    card_path = Path(__file__).parent / "evaluator" / "green_agent.toml"
    agent_card = load_agent_card_toml(str(card_path))
    return JSONResponse(agent_card)


async def handle_message(request: Request) -> JSONResponse:
    """Handle incoming A2A message.
    
    Args:
        request: HTTP request with A2A message
        
    Returns:
        JSON response with evaluation results
    """
    try:
        data = await request.json()
        
        # Extract message content
        message = data.get("message", {})
        parts = message.get("parts", [])
        
        # Get text content
        text_content = ""
        for part in parts:
            if part.get("kind") == "text":
                text_content = part.get("text", "")
                break
        
        # Parse request (expecting JSON in text)
        try:
            request_data = json.loads(text_content)
        except json.JSONDecodeError:
            # If not JSON, treat as simple challenge request
            request_data = {
                "challenges": [text_content],
                "agent_config": {"mode": "internal"}
            }
        
        # Execute evaluation
        executor = GreenAgentExecutor()
        result = await executor.execute(request_data)
        
        # Format A2A response
        context_id = message.get("contextId", "")
        task_id = message.get("taskId", "")
        message_id = message.get("messageId", "")
        
        response = {
            "result": {
                "role": "agent",
                "parts": [{
                    "kind": "text",
                    "text": json.dumps(result, indent=2)
                }],
                "messageId": f"response-{message_id}",
                "contextId": context_id,
                "taskId": task_id
            }
        }
        
        return JSONResponse(response)
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "failed"
        }, status_code=500)


def create_app(writeups_path: str = "./data/agentbeats") -> Starlette:
    """Create Starlette application for green agent.
    
    Args:
        writeups_path: Path to writeups directory
        
    Returns:
        Starlette app
    """
    if not STARLETTE_AVAILABLE:
        raise ImportError("Starlette is required. Install with: pip install starlette uvicorn")
    
    routes = [
        Route("/", handle_agent_card, methods=["GET"]),
        Route("/messages", handle_message, methods=["POST"]),
    ]
    
    app = Starlette(debug=True, routes=routes)
    return app


def start_green_agent(
    host: str = "localhost",
    port: int = 9001,
    writeups_path: str = "./data/agentbeats"
):
    """Start the green agent A2A server.
    
    Args:
        host: Server host
        port: Server port
        writeups_path: Path to writeups directory
    """
    print("=" * 70)
    print("Starting BraceGreen Evaluator (Green Agent A2A Server)")
    print("=" * 70)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Writeups: {writeups_path}")
    print(f"Agent Card: http://{host}:{port}/")
    print(f"Messages Endpoint: http://{host}:{port}/messages")
    print("=" * 70)
    
    # Discover and list available challenges
    try:
        challenges = discover_all_challenges(writeups_path)
        if challenges:
            print(f"\n📚 Loaded {len(challenges)} CTF challenge(s):")
            for i, challenge in enumerate(sorted(challenges), 1):
                print(f"   {i}. {challenge}")
        else:
            print(f"\n⚠️  No challenges found in {writeups_path}")
            print("   Data will be cloned from repository on first evaluation request")
    except Exception as e:
        print(f"\n⚠️  Could not list challenges: {e}")
        print("   Challenges will be discovered when needed")
    
    print("=" * 70)
    print()
    
    app = create_app(writeups_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_green_agent()
