import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the CTF Solver (white agent).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens in response")
    args = parser.parse_args()

    skill = AgentSkill(
        id="ctf_solver",
        name="CTF Challenge Solver",
        description="Solves capture-the-flag challenges by predicting the next command to execute",
        tags=["security", "penetration-testing", "ctf"],
        examples=[
            "Given a reconnaissance scenario, predict the next command to identify the target",
            "Solve a privilege escalation challenge step by step"
        ],
    )

    agent_card = AgentCard(
        name="BraceGreen CTF Solver",
        description="An agent that solves CTF challenges by predicting optimal commands",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text", "text/plain"],
        default_output_modes=["text", "text/plain"],
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=True,
            state_transition_history=False
        ),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        ),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

