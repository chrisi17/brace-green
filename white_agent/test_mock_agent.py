#!/usr/bin/env python3
"""Test script for the mock agent."""

import asyncio
from mock_agent import MockAgent


async def test_mock_agent():
    """Test the mock agent with sample contexts."""
    
    # Test for each task mode
    for task_mode in ["command", "anticipated_result", "goal"]:
        print(f"\n{'='*70}")
        print(f"Testing task_mode: {task_mode}")
        print('='*70)
        
        agent = MockAgent(task_mode=task_mode)
        
        # Test cases for different challenges and steps
        test_cases = [
            {
                "context": "Challenge: Funbox\nStep 1: Identify the target IP address on the network",
                "expected": "Should return answer for Funbox step 1"
            },
            {
                "context": "Challenge: CengBox2\nStep 5: Examine note.txt file",
                "expected": "Should return answer for CengBox2 step 5"
            },
            {
                "context": "Challenge: Victim1\nStep 3: Download the WPA-01.cap file for further analysis",
                "expected": "Should return answer for Victim1 step 3"
            },
            {
                "context": "Challenge: UnknownChallenge\nStep 1: Do something",
                "expected": "Should return 'I don't know'"
            },
            {
                "context": "No challenge or step information here",
                "expected": "Should return 'I don't know'"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['expected']}")
            print(f"Context: {test_case['context'][:80]}...")
            result = await agent.predict(test_case['context'])
            print(f"Result: {result[:100]}{'...' if len(result) > 100 else ''}")


if __name__ == "__main__":
    asyncio.run(test_mock_agent())

