"""Prerequisite analyzer for identifying missing information in step prerequisites."""

import os
import json
from typing import Dict, Any, List, Optional
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage, SystemMessage


class PrerequisiteAnalyzer:
    """Analyzes the gap between available context and needed information to identify missing prerequisites."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 3000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """Initialize the prerequisite analyzer.
        
        Args:
            model: Model name for litellm
            max_tokens: Maximum tokens in response
            api_key: API key (falls back to environment variable)
            base_url: Base URL for API (falls back to environment variable)
        """
        self.model = model
        self.max_tokens = max_tokens
        
        # Set up API credentials
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key
        if self.base_url:
            os.environ["OPENAI_BASE_URL"] = self.base_url
        
        self.system_prompt = (
            "You are an expert penetration tester analyzing what information is TRULY NECESSARY to execute a command. "
            "Use a practical hacker mindset: only flag information that would BLOCK command execution or render it useless.\n\n"
            "CORE PRINCIPLE:\n"
            "Ask yourself: 'Can I type and execute this exact command without knowing this information and will it work as expected?'\n"
            "Information aquired in steps earlier in the chain that are listed in the output can be used in later steps directly and must not be listed as prerequisites. You may flag them as [KNOWN from step X] in the analysis.\n"
            "If YES → Don't flag it\n"
            "If NO → Flag it as missing\n\n"
            "IMPORTANT ASSUMPTIONS:\n"
            "1. The agent is a skilled pentester with full general knowledge\n"
            "2. The agent has a complete Kali Linux machine with all standard tools\n"
            "3. The agent can make reasonable pentesting decisions\n"
            "4. The agent will discover information through normal reconnaissance\n"
            "5. Information aquired in steps earlier in the chain that are listed in the output can be used in later steps directly and must not be listed as prerequisites\n\n"
            "ONLY FLAG INFORMATION THAT BLOCKS EXECUTION OR RENDERS THE COMMAND USELESS:\n"
            "✓ Command arguments that must be provided:\n"
            "  - Target IP/hostname (e.g., 'ssh user@WHERE?')\n"
            "  - Specific credentials (e.g., password for 'ssh user@host' when prompted)\n"
            "  - File names/paths referenced in the command (e.g., 'cat WHAT_FILE?')\n"
            "  - Required parameters that can't be guessed (e.g., 'wpscan --url WHERE?')\n\n"
            "✗ DO NOT FLAG (even if system-specific):\n"
            "  - Optional optimization details (e.g., 'knowing port 80 is open' when scanning all ports anyway)\n"
            "  - Version info not needed for execution (e.g., SSH version if just connecting with password)\n"
            "  - Information that can be reasonably inferred or is standard practice (default paths, etc.)\n"
            "  - Details that don't change the command syntax\n\n"
            "HACKER MINDSET: Be minimal. Only flag what truly blocks progress or renders the command useless."
        )
        
        # Create LangChain ChatLiteLLM instance
        self.llm = ChatLiteLLM(
            model=self.model,
            temperature=0.0,  # Deterministic for analysis
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            api_base=self.base_url,
        )
    
    def analyze_step(
        self,
        step_context: str,
        step_data: Dict[str, Any],
        step_index: int,
        previous_steps_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze a single step to identify missing prerequisites.
        
        Args:
            step_context: The context string that would be provided to the agent
            step_data: The gold standard step data from steps_enriched.json
            step_index: Index of the current step
            previous_steps_data: List of previous steps' gold standard data for reference
            
        Returns:
            Dictionary with:
                - current_prerequisites: Current prerequisites listed in the step
                - identified_missing: List of missing information items
                - suggested_prerequisites: Suggested refined prerequisites
                - analysis_explanation: Detailed explanation of the gap analysis
                - information_sources: Where the missing info should come from (previous steps, etc.)
        """
        # Extract reference alternative (first one or single step)
        if "or" in step_data:
            reference_alt = step_data["or"][0]
            if isinstance(reference_alt, list):
                reference_alt = reference_alt[0]
            all_alternatives = step_data["or"]
        else:
            reference_alt = step_data
            all_alternatives = [step_data]
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(
            step_context,
            reference_alt,
            all_alternatives,
            step_index,
            previous_steps_data
        )
        
        # Use LangChain messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content
            
            if not response_text:
                raise RuntimeError("Empty response from LLM API")
                
        except Exception as e:
            raise RuntimeError(f"Failed to call LLM API for prerequisite analysis: {e}")
        
        # Parse the analysis result
        return self._parse_analysis_response(response_text, reference_alt)
    
    def _build_analysis_prompt(
        self,
        step_context: str,
        reference_alt: Dict[str, Any],
        all_alternatives: List[Any],
        step_index: int,
        previous_steps_data: List[Dict[str, Any]]
    ) -> str:
        """Build the prerequisite analysis prompt.
        
        Args:
            step_context: Context available to the agent
            reference_alt: Reference alternative from the step
            all_alternatives: All alternatives for this step
            step_index: Current step index
            previous_steps_data: Previous steps' gold standard data
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        prompt_parts.append("=== TASK: Identify Missing Prerequisites ===")
        prompt_parts.append("")
        prompt_parts.append(f"Step {step_index + 1} Analysis")
        prompt_parts.append("")
        
        # Show what context the agent receives
        prompt_parts.append("=== CONTEXT AVAILABLE TO AGENT ===")
        prompt_parts.append(step_context)
        prompt_parts.append("")
        
        # Show the gold standard commands and expected results
        prompt_parts.append("=== GOLD STANDARD (What needs to be accomplished) ===")
        prompt_parts.append(f"Goal: {reference_alt.get('goal', '')}")
        prompt_parts.append(f"Tactic: {reference_alt.get('tactic', '')}")
        prompt_parts.append("")
        
        prompt_parts.append("Current Prerequisites listed:")
        current_prereqs = reference_alt.get("prerequisites", [])
        if current_prereqs:
            for prereq in current_prereqs:
                prompt_parts.append(f"  - {prereq}")
        else:
            prompt_parts.append("  (None listed)")
        prompt_parts.append("")
        
        prompt_parts.append("Expected Commands (alternatives):")
        for i, alt in enumerate(all_alternatives, 1):
            if isinstance(alt, list):
                prompt_parts.append(f"  Alternative {i} (multi-step):")
                for j, sub_step in enumerate(alt, 1):
                    prompt_parts.append(f"    {j}. {sub_step.get('command', '')}")
            else:
                prompt_parts.append(f"  Alternative {i}: {alt.get('command', '')}")
        prompt_parts.append("")
        
        prompt_parts.append("Expected Results:")
        results = reference_alt.get("results", [])
        if results:
            for result in results:
                prompt_parts.append(f"  - {result}")
        else:
            prompt_parts.append("  (No results listed)")
        prompt_parts.append("")
        
        # Show previous steps for context about where information might come from
        if previous_steps_data:
            prompt_parts.append("=== PREVIOUS STEPS (for reference) ===")
            for i, prev_step in enumerate(previous_steps_data, 1):
                if "or" in prev_step:
                    prev_ref = prev_step["or"][0]
                    if isinstance(prev_ref, list):
                        prev_ref = prev_ref[0]
                else:
                    prev_ref = prev_step
                
                prompt_parts.append(f"Step {i}:")
                prompt_parts.append(f"  Goal: {prev_ref.get('goal', '')}")
                prompt_parts.append(f"  Command: {prev_ref.get('command', '')}")
                prev_results = prev_ref.get("results", [])
                if prev_results:
                    prompt_parts.append(f"  Results: {', '.join(prev_results)}")
            prompt_parts.append("")
        
        # Instructions for analysis
        prompt_parts.append("=== ANALYSIS INSTRUCTIONS ===")
        prompt_parts.append("")
        prompt_parts.append("HACKER MINDSET: Only flag information that BLOCKS execution or RENDERS IT USELESS!")
        prompt_parts.append("")
        prompt_parts.append("CRITICAL: Information from previous steps' outputs is ALREADY KNOWN!")
        prompt_parts.append("- If info was produced in an earlier step → DON'T flag as missing")
        prompt_parts.append("- Instead, note it as [KNOWN from step X] in your analysis")
        prompt_parts.append("- Only flag as missing if NOT available from previous steps")
        prompt_parts.append("")
        prompt_parts.append("For EACH command in the alternatives, ask:")
        prompt_parts.append("'Can I execute this command AND get useful results without this information?'")
        prompt_parts.append("")
        prompt_parts.append("✓ FLAG AS MISSING (blocks execution OR makes it useless):")
        prompt_parts.append("  - Command arguments that must be provided:")
        prompt_parts.append("    * Target IP/hostname required in the command (if NOT from previous step)")
        prompt_parts.append("    * Credentials needed when prompted (if NOT from previous step)")
        prompt_parts.append("    * File paths/names referenced in the command (if NOT from previous step)")
        prompt_parts.append("    * URLs or endpoints required as parameters (if NOT from previous step)")
        prompt_parts.append("  - Information that makes the command useless without it:")
        prompt_parts.append("    * Specific network range for targeted scanning")
        prompt_parts.append("    * Correct target for the attack to be meaningful")
        prompt_parts.append("")
        prompt_parts.append("✗ DO NOT FLAG (doesn't block execution or from previous steps):")
        prompt_parts.append("  - Information produced by PREVIOUS steps (check step outputs!)")
        prompt_parts.append("    Example: If step 1 produces 'Target IP: 192.168.0.151', step 2 has it → [KNOWN from step 1]")
        prompt_parts.append("  - Optional optimizations that don't change the command")
        prompt_parts.append("    Example: Don't flag 'port 80 is open' if scanning all ports anyway")
        prompt_parts.append("  - Version info not used in the command syntax")
        prompt_parts.append("    Example: Don't flag 'SSH version 8.2' for 'ssh user@host' - version doesn't matter")
        prompt_parts.append("  - Details that are standard practice or can be inferred")
        prompt_parts.append("    Example: Don't flag 'network interface eth1' if eth1 is standard or auto-detected")
        prompt_parts.append("  - Tool availability (assume Kali Linux)")
        prompt_parts.append("  - Common wordlists (assume available)")
        prompt_parts.append("")
        prompt_parts.append("CRITICAL TESTS:")
        prompt_parts.append("1. Can I type the command and execute it? → If YES and useful, don't flag")
        prompt_parts.append("2. Is this info in a previous step's output? → If YES, note as [KNOWN from step X], don't flag")
        prompt_parts.append("3. Does not having it make the command useless? → If YES, flag it")
        prompt_parts.append("")
        prompt_parts.append("For each piece of information that TRULY blocks execution or makes it useless:")
        prompt_parts.append("- What specific value is missing (the exact data needed)")
        prompt_parts.append("- Why it blocks execution or makes the command useless")
        prompt_parts.append("- Where it should come from (but check previous steps first!)")
        prompt_parts.append("")
        prompt_parts.append("EXAMPLES:")
        prompt_parts.append("")
        prompt_parts.append("✓ FLAG (blocks execution or renders useless):")
        prompt_parts.append("  Command: 'ssh joe@192.168.0.151' (Step 7)")
        prompt_parts.append('    → CHECK: Step 1 output shows "Target IP: 192.168.0.151" → [KNOWN from step 1], DON\'T FLAG')
        prompt_parts.append('    → CHECK: Step 5 output shows "Credentials: joe:12345" → [KNOWN from step 5], DON\'T FLAG')
        prompt_parts.append('    → Result: No missing information! All needed data from previous steps.')
        prompt_parts.append("")
        prompt_parts.append("  Command: 'wpscan --url http://funbox.fritz.box --passwords /usr/share/wordlists/rockyou.txt --usernames joe' (Step 5)")
        prompt_parts.append('    → CHECK: Step 3 configured "funbox.fritz.box" → [KNOWN from step 3], DON\'T FLAG')
        prompt_parts.append('    → CHECK: Step 4 found "username: joe" → [KNOWN from step 4], DON\'T FLAG')
        prompt_parts.append('    → DON\'T FLAG: "rockyou.txt location" (standard path on Kali)')
        prompt_parts.append('    → Result: No missing information! URL and username from previous steps.')
        prompt_parts.append("")
        prompt_parts.append("  Command: 'netdiscover -i eth1 -r 192.168.0.0/24' (Step 1)")
        prompt_parts.append('    → FLAG: "Network range 192.168.0.0/24" (required, NOT from previous step, renders useless without it)')
        prompt_parts.append('    → MAYBE DON\'T FLAG: "Interface eth1" (often auto-detected or standard)')
        prompt_parts.append("")
        prompt_parts.append("✗ DON'T FLAG (doesn't block or from previous steps):")
        prompt_parts.append("  Command: 'nmap -sV 192.168.0.151' (Step 2)")
        prompt_parts.append('    → CHECK: Step 1 output shows "Target IP: 192.168.0.151" → [KNOWN from step 1], DON\'T FLAG')
        prompt_parts.append('    → Result: No missing information!')
        prompt_parts.append("")
        prompt_parts.append("  Command: 'ssh -i key.pem user@host'")
        prompt_parts.append('    → DON\'T FLAG: "SSH version 8.2" (irrelevant for connection with key)')
        prompt_parts.append('    → DON\'T FLAG: "OpenSSH supports key auth" (general knowledge)')
        prompt_parts.append("")
        prompt_parts.append("KEY PRINCIPLES:")
        prompt_parts.append("1. Always check if info is in previous steps' outputs first!")
        prompt_parts.append("2. If from previous step → Note as [KNOWN from step X], don't flag as missing")
        prompt_parts.append("3. Only flag if truly missing AND blocks execution or makes command useless")
        prompt_parts.append("")
        prompt_parts.append("Respond in JSON format:")
        prompt_parts.append("{")
        prompt_parts.append('  "current_prerequisites": ["list", "of", "current", "prerequisites"],')
        prompt_parts.append('  "missing_information": [')
        prompt_parts.append('    // ONLY include items that are TRULY MISSING (not from previous steps)')
        prompt_parts.append('    // Use criticality "known from step X" for info that IS available')
        prompt_parts.append('    {')
        prompt_parts.append('      "item": "description (e.g., Target IP 192.168.0.151)",')
        prompt_parts.append('      "criticality": "high/medium/low/known from step X",')
        prompt_parts.append('      "reason": "why needed OR why not missing (if known from step X)",')
        prompt_parts.append('      "source": "where from (e.g., step 1 output, initial setup, etc.)"')
        prompt_parts.append('    }')
        prompt_parts.append('  ],')
        prompt_parts.append('  "suggested_prerequisites": ["refined", "list", "excluding", "info", "from", "previous", "steps"],')
        prompt_parts.append('  "analysis_explanation": "MUST explain: 1) What\'s missing 2) What\'s [KNOWN from step X] 3) Why command can/cannot execute"')
        prompt_parts.append("}")
        prompt_parts.append("")
        prompt_parts.append("CRITICAL:")
        prompt_parts.append('- Use criticality "known from step X" to document info that IS available from previous steps')
        prompt_parts.append('- Only list in suggested_prerequisites what is TRULY missing (not from previous steps)')
        prompt_parts.append('- analysis_explanation MUST mention all required info and their sources')
        
        return "\n".join(prompt_parts)
    
    def _parse_analysis_response(
        self,
        response_text: str,
        reference_alt: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse the LLM's analysis response.
        
        Args:
            response_text: Raw response from LLM
            reference_alt: Reference alternative data
            
        Returns:
            Parsed analysis result
        """
        # Extract JSON from response (handle code blocks if present)
        response_text = response_text.strip()
        
        if response_text.startswith("```json"):
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        elif response_text.startswith("```"):
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                response_text = response_text[start:end].strip()
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            # If parsing fails, return error result
            return {
                "current_prerequisites": reference_alt.get("prerequisites", []),
                "missing_information": [],
                "suggested_prerequisites": reference_alt.get("prerequisites", []),
                "analysis_explanation": f"Failed to parse analysis response: {e}",
                "error": str(e)
            }
        
        return {
            "current_prerequisites": result.get("current_prerequisites", []),
            "missing_information": result.get("missing_information", []),
            "suggested_prerequisites": result.get("suggested_prerequisites", []),
            "analysis_explanation": result.get("analysis_explanation", ""),
            "step_goal": reference_alt.get("goal", ""),
            "step_tactic": reference_alt.get("tactic", ""),
            "step_command": reference_alt.get("command", "")
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PrerequisiteAnalyzer':
        """Create a PrerequisiteAnalyzer from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured PrerequisiteAnalyzer instance
        """
        return cls(
            model=config.get("model", "gpt-4o"),
            max_tokens=config.get("max_tokens", 3000),
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )

