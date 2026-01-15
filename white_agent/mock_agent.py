"""Mock agent that deterministically replays answers from evaluation results.

This module provides a mock agent for testing that extracts challenge and step
information from the green agent's context message and returns pre-recorded
answers from evaluation results.
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List


class MockAgent:
    """Mock agent that replays answers from evaluation result files."""
    
    def __init__(self, task_mode: str = "command"):
        """Initialize the mock agent.
        
        Args:
            task_mode: Task mode (command, anticipated_result, or goal)
        """
        self.task_mode = task_mode
        self.mock_data = self._load_mock_data()
        
    def _load_mock_data(self) -> Dict[str, Any]:
        """Load mock evaluation results from JSON files.
        
        Returns:
            Dictionary mapping challenge names to their evaluation results
        """
        # Load from mock_data directory within white_agent
        data_dir = Path(__file__).parent / "mock_data"
        mock_data = {}
        
        # Load evaluation results for each task mode
        result_files = {
            "command": "LSX-UniWue-20260115-172731.json",
            "anticipated_result": "LSX-UniWue-20260115-172655.json",
            "goal": "LSX-UniWue-20260115-173651.json"
        }
        
        result_file = result_files.get(self.task_mode)
        if result_file:
            result_path = data_dir / result_file
            if result_path.exists():
                with open(result_path, 'r') as f:
                    results = json.load(f)
                    # Extract results by challenge
                    if "results" in results and len(results["results"]) > 0:
                        for challenge_result in results["results"][0]["results"]:
                            challenge_name = challenge_result["challenge"]
                            mock_data[challenge_name] = challenge_result
        
        return mock_data
    
    def _extract_challenge_name(self, context: str) -> Optional[str]:
        """Extract challenge name from context message.
        
        Args:
            context: Context message from green agent
            
        Returns:
            Challenge name if found, None otherwise
        """
        # Look for challenge names in the context
        challenge_patterns = [
            r"Challenge:\s*([A-Za-z0-9_-]+)",
            r"challenge\s+([A-Za-z0-9_-]+)",
            r"CTF:\s*([A-Za-z0-9_-]+)",
        ]
        
        for pattern in challenge_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Try to match known challenge names directly
        known_challenges = ["CengBox2", "Funbox", "Victim1"]
        for challenge in known_challenges:
            if challenge.lower() in context.lower():
                return challenge
        
        return None
    
    def _extract_step_number(self, context: str) -> Optional[int]:
        """Extract step number from context message.
        
        Args:
            context: Context message from green agent
            
        Returns:
            Step number (0-indexed) if found, None otherwise
        """
        # Look for step indicators
        step_patterns = [
            r"Current Step:\s+(\d+)",  # Matches "Current Step: 1 of 16"
            r"Step:\s+(\d+)",           # Matches "Step: 1"
            r"Step\s+(\d+)",            # Matches "Step 1"
            r"step\s+#?(\d+)",          # Matches "step #1" or "step 1"
            r"Iteration\s+(\d+)",       # Matches "Iteration 1"
        ]
        
        for pattern in step_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                # Convert to 0-indexed
                return int(match.group(1)) - 1
        
        return None
    
    def _find_matching_answer(
        self, 
        challenge_name: str, 
        step_number: int
    ) -> Optional[str]:
        """Find the matching answer for a challenge and step.
        
        Args:
            challenge_name: Name of the challenge
            step_number: Step number (0-indexed)
            
        Returns:
            Matched answer string if found, None otherwise
        """
        if challenge_name not in self.mock_data:
            return None
        
        challenge_data = self.mock_data[challenge_name]
        steps = challenge_data.get("steps_completed", [])
        
        if step_number < 0 or step_number >= len(steps):
            return None
        
        step = steps[step_number]
        
        # Determine the field to extract based on task mode
        field_map = {
            "command": "matched_command",
            "anticipated_result": "matched_anticipated_result",
            "goal": "matched_goal"
        }
        
        field_name = field_map.get(self.task_mode)
        if not field_name:
            return None
        
        # Handle nested "or" structures
        if "or" in step:
            # Find the first completed alternative
            for alternative in step["or"]:
                if isinstance(alternative, dict) and alternative.get("completed"):
                    return alternative.get(field_name)
                elif isinstance(alternative, list):
                    # Handle nested lists
                    for nested_alt in alternative:
                        if isinstance(nested_alt, dict) and nested_alt.get("completed"):
                            return nested_alt.get(field_name)
        else:
            # Simple step
            if step.get("completed"):
                return step.get(field_name)
        
        return None
    
    async def predict(self, context: str) -> str:
        """Generate a prediction based on the context.
        
        Args:
            context: Context message from green agent
            
        Returns:
            Predicted answer or "I don't know"
        """
        # Extract challenge and step information
        challenge_name = self._extract_challenge_name(context)
        step_number = self._extract_step_number(context)
        
        if challenge_name is None or step_number is None:
            return "I don't know"
        
        # Find matching answer
        answer = self._find_matching_answer(challenge_name, step_number)
        
        if answer:
            return answer
        else:
            return "I don't know"

