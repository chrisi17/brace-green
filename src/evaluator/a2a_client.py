"""A2A client utilities for communicating with remote agents.

This module provides utilities for sending messages to A2A-compatible agents
and parsing their responses, with support for context management across
multi-turn conversations.
"""

import httpx
import json
from typing import Optional, Dict, Any
from uuid import uuid4


class A2AClient:
    """Client for communicating with A2A-compatible agents."""
    
    def __init__(self, agent_url: str, timeout: float = 300.0):
        """Initialize A2A client.
        
        Args:
            agent_url: Base URL of the A2A agent
            timeout: Request timeout in seconds (default: 300s for long-running evals)
        """
        self.agent_url = agent_url.rstrip('/')
        self.timeout = timeout
        self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
    
    async def send_message(
        self,
        message: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a message to the A2A agent and get response.
        
        Args:
            message: Text message to send to the agent
            context_id: Optional context ID for multi-turn conversations
            task_id: Optional task ID
            
        Returns:
            Response dictionary containing the agent's reply
            
        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If the response format is invalid
        """
        if not self._client:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        
        # Generate IDs if not provided
        message_id = uuid4().hex
        if not task_id:
            task_id = uuid4().hex
        if not context_id:
            context_id = uuid4().hex
        
        # Construct A2A message payload
        payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            }
        }
        
        # Send request to A2A endpoint
        url = f"{self.agent_url}/messages"
        
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            
            response_data = response.json()
            return {
                "content": self._extract_text_content(response_data),
                "context_id": context_id,
                "task_id": task_id,
                "full_response": response_data
            }
            
        except httpx.HTTPError as e:
            raise httpx.HTTPError(f"A2A request failed: {e}")
    
    def _extract_text_content(self, response_data: Dict[str, Any]) -> str:
        """Extract text content from A2A response.
        
        Args:
            response_data: Raw A2A response data
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If response format is invalid
        """
        try:
            # Handle different A2A response formats
            if "result" in response_data:
                result = response_data["result"]
                
                # Check for message result
                if isinstance(result, dict) and "parts" in result:
                    parts = result["parts"]
                    text_parts = [
                        part.get("text", "")
                        for part in parts
                        if part.get("kind") == "text"
                    ]
                    return "\n".join(text_parts) if text_parts else ""
                
                # Check for artifact result
                if isinstance(result, dict) and "artifact" in result:
                    artifact = result["artifact"]
                    if isinstance(artifact, dict) and "parts" in artifact:
                        parts = artifact["parts"]
                        text_parts = [
                            part.get("text", "")
                            for part in parts
                            if part.get("kind") == "text"
                        ]
                        return "\n".join(text_parts) if text_parts else ""
            
            # Fallback: try to extract content field
            if "content" in response_data:
                content = response_data["content"]
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict):
                    return json.dumps(content)
            
            # If we can't find content, return the whole response as JSON
            return json.dumps(response_data)
            
        except Exception as e:
            raise ValueError(f"Failed to extract content from A2A response: {e}")
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


def parse_agent_response(response: Dict[str, Any]) -> str:
    """Parse A2A agent response to extract prediction.
    
    This is a convenience function for extracting the agent's prediction
    from the response dictionary returned by send_message().
    
    Args:
        response: Response dictionary from send_message()
        
    Returns:
        Extracted prediction text
    """
    content = response.get("content", "")
    
    # Clean up the response - remove any markdown code blocks, extra whitespace
    content = content.strip()
    
    # Remove common markdown code block patterns
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (opening ```)
        lines = lines[1:]
        # Remove last line if it's closing ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    
    return content

