"""White agent A2A server - CTF solving agent as A2A service.

This module provides an A2A-compatible server that wraps the LLM agent,
allowing it to be called remotely by evaluators or other systems.
"""

import uvicorn
import tomllib
import os
import logging
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
from starlette.requests import Request

from .evaluator.agent_interface import LLMAgentInterface

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class WhiteAgentExecutor:
    """Executor for white agent (CTF solver) requests."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 500,
        api_key: str = None,
        base_url: str = None
    ):
        """Initialize the white agent executor.
        
        Args:
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: API key (falls back to environment)
            base_url: Base URL for API (falls back to environment)
        """
        self.agent = LLMAgentInterface(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model
    
    async def execute(self, message: str, context_id: str) -> str:
        """Execute a prediction request.
        
        Args:
            message: Context message from evaluator
            context_id: Context ID for conversation tracking
            
        Returns:
            Predicted command
        """
        # The agent's predict_next_step expects the full context
        prediction = self.agent.predict_next_step(message)
        return prediction


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
    card_path = Path(__file__).parent / "evaluator" / "white_agent.toml"
    agent_card = load_agent_card_toml(str(card_path))
    return JSONResponse(agent_card)


async def handle_message(request: Request) -> JSONResponse:
    """Handle incoming A2A message.
    
    Args:
        request: HTTP request with A2A message
        
    Returns:
        JSON response with agent's prediction
    """
    try:
        data = await request.json()
        
        # Extract message content
        message = data.get("message", {})
        parts = message.get("parts", [])
        
        # Get text content (the context from evaluator)
        text_content = ""
        for part in parts:
            if part.get("kind") == "text":
                text_content = part.get("text", "")
                break
        
        if not text_content:
            return JSONResponse({
                "error": "No text content in message",
                "status": "failed"
            }, status_code=400)
        
        # Get context and task IDs
        context_id = message.get("contextId", "")
        task_id = message.get("taskId", "")
        message_id = message.get("messageId", "")
        
        # Get executor from app state
        executor = request.app.state.executor
        
        # Log incoming task
        logger.info("=" * 70)
        logger.info("WHITE AGENT: Prompted with task:")
        logger.info("-" * 70)
        logger.info(text_content)
        logger.info("-" * 70)
        
        # Get prediction from agent
        prediction = await executor.execute(text_content, context_id)
        
        # Log response
        logger.info("WHITE AGENT: Returning answer:")
        logger.info("-" * 70)
        logger.info(prediction)
        logger.info("=" * 70)
        logger.info("")
        
        # Format A2A response
        response = {
            "result": {
                "role": "agent",
                "parts": [{
                    "kind": "text",
                    "text": prediction
                }],
                "messageId": f"response-{message_id}",
                "contextId": context_id,
                "taskId": task_id
            }
        }
        
        return JSONResponse(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "error": str(e),
            "status": "failed"
        }, status_code=500)


def create_app(
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 500,
    api_key: str = None,
    base_url: str = None
) -> Starlette:
    """Create Starlette application for white agent.
    
    Args:
        model: LLM model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        api_key: API key
        base_url: Base URL for API
        
    Returns:
        Starlette app
    """
    routes = [
        Route("/", handle_agent_card, methods=["GET"]),
        Route("/.well-known/agent-card.json", handle_agent_card, methods=["GET"]),
        Route("/messages", handle_message, methods=["POST"]),
    ]
    
    app = Starlette(debug=True, routes=routes)
    
    # Store executor in app state
    app.state.executor = WhiteAgentExecutor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url
    )
    
    return app


def start_white_agent(
    host: str = "localhost",
    port: int = 8000,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 500,
    api_key: str = None,
    base_url: str = None
):
    """Start the white agent A2A server.
    
    Args:
        host: Server host
        port: Server port
        model: LLM model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        api_key: API key
        base_url: Base URL for API
    """
    print("=" * 70)
    print("BraceGreen CTF Solving Agent (White Agent A2A Server)")
    print("=" * 70)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Agent Card: http://{host}:{port}/")
    print(f"Messages Endpoint: http://{host}:{port}/messages")
    print("=" * 70)
    print()
    
    app = create_app(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url
    )
    
    # Reduce uvicorn access log verbosity
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    uvicorn_log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"
    
    uvicorn.run(app, host=host, port=port, log_config=uvicorn_log_config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run white agent (CTF solver) as A2A server"
    )
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="API base URL")
    
    args = parser.parse_args()
    
    start_white_agent(
        host=args.host,
        port=args.port,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        base_url=args.base_url
    )

