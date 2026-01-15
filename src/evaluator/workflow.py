"""LangGraph workflow for evaluating red-teaming agents."""

import os
from typing import Dict, Any, Literal, Optional, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from .state import EvaluationState
from .agent_interface import LLMAgentInterface
from .step_evaluator import StepEvaluator
from .utils import (
    build_step_context,
    format_evaluation_results,
    load_challenge_steps
)

# Phoenix imports (optional - only if available)
try:
    from phoenix.otel import register
    from opentelemetry import trace
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    register = None
    trace = None

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class EvaluatorWorkflow:
    """LangGraph workflow for step-by-step agent evaluation."""
    
    def __init__(
        self,
        agent_interface: LLMAgentInterface,
        step_evaluator: StepEvaluator,
        max_iterations_per_step: int = 10,
        enable_phoenix: bool = False,
        include_goal: str = "first",
        include_tactic: str = "first",
        include_prerequisites: str = "always",
        history_context: List[str] = None
    ):
        """Initialize the evaluator workflow.
        
        Args:
            agent_interface: Interface to the agent under test
            step_evaluator: Evaluator for comparing predictions
            max_iterations_per_step: Maximum iterations per step
            enable_phoenix: Whether to enable Phoenix tracing (default: True)
            include_goal: When to include goals in prompts: "first", "always", or "never" (default: "first")
            include_tactic: When to include tactics in prompts: "first", "always", or "never" (default: "first")
            include_prerequisites: When to include prerequisites in prompts: "first", "always", or "never" (default: "always")
            history_context: Fields to include in context history: "goal", "command", "output", "results" 
                           (default: all fields)
        """
        self.agent_interface = agent_interface
        self.step_evaluator = step_evaluator
        self.max_iterations_per_step = max_iterations_per_step
        self.enable_phoenix = enable_phoenix and PHOENIX_AVAILABLE
        self.include_goal = include_goal
        self.include_tactic = include_tactic
        self.include_prerequisites = include_prerequisites
        self.history_context = history_context if history_context is not None else ["goal", "command", "output", "results"]
        
        # Store evaluation protocol for potential use
        self.evaluation_protocol = step_evaluator.evaluation_protocol
        
        # Check if Phoenix is already initialized (should be done in main.py)
        # If already initialized, the tracer provider will be active
        self.phoenix_tracer = None
        if self.enable_phoenix:
            try:
                # Check if tracer provider is already set up
                if trace and trace.get_tracer_provider().__class__.__name__ != 'ProxyTracerProvider':
                    # Phoenix already initialized in main.py
                    print("✓ Phoenix tracing active (initialized before workflow creation)")
                    self.phoenix_tracer = trace.get_tracer_provider()
                else:
                    # Fallback: initialize here if not done earlier (not recommended)
                    print(f"{Colors.YELLOW}⚠ Phoenix not initialized early - initializing now (may affect trace context){Colors.RESET}")
                    self._init_phoenix()
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Warning: Phoenix check failed: {e}{Colors.RESET}")
                self.enable_phoenix = False
                self.phoenix_tracer = None
        
        # Build the main workflow graph
        self.graph = self._build_main_graph()
        
        # Build the step evaluation subgraph
        self.step_subgraph = self._build_step_subgraph()
    
    def _init_phoenix(self) -> None:
        """Initialize Phoenix for observability and tracing (fallback method).
        
        This is a fallback in case Phoenix wasn't initialized in main.py.
        It's recommended to initialize Phoenix in main.py before creating the workflow.
        """
        try:
            project_name = os.getenv("PHOENIX_PROJECT_NAME", "bracegreen-evaluator")
            
            # Register Phoenix tracer with auto-instrumentation
            self.phoenix_tracer = register(
                project_name=project_name,
                auto_instrument=True
            )
            
            print("✓ Phoenix tracing enabled (fallback initialization)")
            print(f"  Project: {project_name}")
            
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Warning: Could not initialize Phoenix tracing: {e}{Colors.RESET}")
            print(f"{Colors.YELLOW}  Continuing without Phoenix tracing...{Colors.RESET}")
            self.enable_phoenix = False
            self.phoenix_tracer = None
    
    def _build_main_graph(self) -> StateGraph:
        """Build the main evaluation workflow graph.
        
        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(EvaluationState)
        
        # Add nodes
        workflow.add_node("load_challenge", self._load_challenge_node)
        workflow.add_node("prepare_step", self._prepare_step_node)
        workflow.add_node("evaluate_step", self._evaluate_step_wrapper)
        workflow.add_node("record_result", self._record_result_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.add_edge(START, "load_challenge")
        workflow.add_edge("load_challenge", "prepare_step")
        workflow.add_edge("prepare_step", "evaluate_step")
        workflow.add_edge("evaluate_step", "record_result")
        workflow.add_conditional_edges(
            "record_result",
            self._check_more_steps,
            {
                "continue": "prepare_step",
                "done": "finalize"
            }
        )
        workflow.add_edge("finalize", END)
        
        # Compile with memory for checkpointing and higher recursion limit
        return workflow.compile(
            checkpointer=MemorySaver(),
            debug=False
        )
    
    def _build_step_subgraph(self) -> StateGraph:
        """Build the step evaluation subgraph.
        
        This subgraph iteratively prompts the agent and evaluates
        responses until the goal is reached.
        
        Returns:
            Compiled StateGraph for step evaluation
        """
        subgraph = StateGraph(EvaluationState)
        
        # Add nodes
        subgraph.add_node("prompt_agent", self._prompt_agent_node)
        subgraph.add_node("evaluate_response", self._evaluate_response_node)
        
        # Add edges
        subgraph.add_edge(START, "prompt_agent")
        subgraph.add_edge("prompt_agent", "evaluate_response")
        subgraph.add_conditional_edges(
            "evaluate_response",
            self._check_goal_reached,
            {
                "continue": "prompt_agent",
                "done": END
            }
        )
        
        return subgraph.compile(debug=False)
    
    # Main graph nodes
    
    def _load_challenge_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Load challenge steps and initialize state.
        
        Args:
            state: Current state
            
        Returns:
            Updated state fields
        """
        print(f"\n=== Loading challenge: {state['challenge_name']} ===\n")
        
        steps = load_challenge_steps(state["challenge_name"])
        
        # Initialize with empty results - we'll build them stepwise
        return {
            "steps": steps,
            "completed_results": [],
            "current_step_index": 0,
            "agent_predictions": [],
            "context_history": [],
            "current_iteration": 0,
            "current_step_goal_reached": False
        }
    
    def _prepare_step_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Prepare for evaluating the next step.
        
        Args:
            state: Current state
            
        Returns:
            Updated state fields
        """
        step_index = state["current_step_index"]
        task_mode = state.get("task_mode", "command")
        include_goal = state.get("include_goal", "always")
        
        # Special case: Skip evaluation of step 0 when task_mode=goal and include_goal=first
        # This step serves as an example - show what a goal looks like
        if step_index == 0 and task_mode == "goal" and include_goal == "first":
            current_step = state["steps"][step_index]
            reference_alt = self._get_gold_alternative(current_step)
            
            # Handle multi-step alternatives (list) vs single step (dict)
            if isinstance(reference_alt, list):
                goal = reference_alt[0].get("goal", "Unknown goal") if reference_alt else "Unknown goal"
            else:
                goal = reference_alt.get("goal", "Unknown goal")
            
            print(f"\n{Colors.BOLD}=== Step {step_index + 1}/{len(state['steps'])} (Example - Not Evaluated) ==={Colors.RESET}")
            print(f"Goal: {goal}")
            print(f"{Colors.YELLOW}⚠ Skipping evaluation of first step (task_mode=goal + include_goal=first){Colors.RESET}")
            print(f"{Colors.YELLOW}  This step will be added to context as an example of what a goal looks like{Colors.RESET}")
            print(f"  Actual evaluation starts from step 2\n")
            
            # Build result for this step (NOT marking as completed)
            step_result = self._build_step_result(current_step, None, task_mode=task_mode)
            # Mark this as an example step that doesn't count toward score
            step_result["_example_step"] = True
            # Mark the gold alternative as NOT completed (it's just an example)
            if "or" in step_result:
                for alt in step_result["or"]:
                    if isinstance(alt, list):
                        if alt and alt[0].get("gold", False):
                            alt[0]["completed"] = False
                            alt[0]["_example_step"] = True
                    elif alt.get("gold", False):
                        alt["completed"] = False
                        alt["_example_step"] = True
            else:
                step_result["completed"] = False
                step_result["_example_step"] = True
            
            # Build context entry for this example step
            history_fields = state.get("history_context", ["goal", "command", "output", "results"])
            context_entry = self._build_context_entry(current_step, None, history_fields)
            new_history = state.get("context_history", []) + [context_entry]
            
            # Move directly to next step
            return {
                "current_step_index": step_index + 1,
                "context_history": new_history,
                "completed_results": state.get("completed_results", []) + [step_result],
                "agent_predictions": [],
                "current_iteration": 0,
                "current_step_goal_reached": False,
                "_is_fine_grained": False,
                "_accumulated_commands": None,
                "_step_eval_result": None
            }
        
        # Extract goal for logging (use gold alternative)
        if step_index < len(state["steps"]):
            current_step = state["steps"][step_index]
            reference_alt = self._get_gold_alternative(current_step)
            # Handle multi-step alternatives (list) vs single step (dict)
            if isinstance(reference_alt, list):
                goal = reference_alt[0].get("goal", "Unknown goal") if reference_alt else "Unknown goal"
                tactic = reference_alt[0].get("tactic", "") if reference_alt else ""
            else:
                goal = reference_alt.get("goal", "Unknown goal")
                tactic = reference_alt.get("tactic", "")
            
            print(f"\n{Colors.BOLD}=== Preparing step {step_index + 1}/{len(state['steps'])} ==={Colors.RESET}")
            print(f"Goal: {goal}")
            if tactic:
                print(f"Tactic: {tactic}")
            print()
        else:
            print(f"\n{Colors.BOLD}=== Preparing step {step_index + 1}/{len(state['steps'])} ==={Colors.RESET}\n")
        
        # Reset for new step
        return {
            "agent_predictions": [],
            "current_iteration": 0,
            "current_step_goal_reached": False,
            "_is_fine_grained": False,
            "_accumulated_commands": None
        }
    
    def _evaluate_step_wrapper(self, state: EvaluationState) -> Dict[str, Any]:
        """Wrapper to run the step evaluation subgraph.
        
        Args:
            state: Current state
            
        Returns:
            Updated state from subgraph execution
        """
        step_index = state["current_step_index"]
        current_step = state["steps"][step_index]
        
        # Extract goal for logging (use gold alternative)
        reference_alt = self._get_gold_alternative(current_step)
        # Handle multi-step alternatives (list) vs single step (dict)
        if isinstance(reference_alt, list):
            goal = reference_alt[0].get("goal", "Unknown goal") if reference_alt else "Unknown goal"
        else:
            goal = reference_alt.get("goal", "Unknown goal")
        
        print(f"{Colors.BOLD}=== Evaluating step {step_index + 1} ==={Colors.RESET}")
        print(f"Goal: {goal}\n")
        
        # Run the step evaluation subgraph with recursion limit
        # Set recursion limit to max_iterations + some buffer for safety
        # Minimum of 100 to ensure subgraph has enough room
        recursion_limit = max(100, state.get("max_iterations_per_step", 10) * 5 + 5)
        
        # Create config with recursion limit
        config = {
            "recursion_limit": recursion_limit
        }
        result = self.step_subgraph.invoke(state, config)
        
        return result
    
    def _record_result_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Record the evaluation result for the current step.
        
        Builds the result structure for this step and appends it to completed_results.
        
        Args:
            state: Current state
            
        Returns:
            Updated state fields
        """
        step_index = state["current_step_index"]
        current_step = state["steps"][step_index]
        
        # Use accumulated commands if available for the matched command
        eval_result = state.get("_step_eval_result")
        if eval_result and state.get("_accumulated_commands"):
            eval_result["matched_command"] = state["_accumulated_commands"]
        
        # Build result structure for this step based on its format
        task_mode = state.get("task_mode", "command")
        step_result = self._build_step_result(current_step, eval_result, task_mode=task_mode)
        
        # Add to completed results
        new_completed_results = state["completed_results"] + [step_result]
        
        # Add to context history for future steps (gold standard path)
        # Use configured history fields to build the context entry
        history_fields = state.get("history_context", ["goal", "command", "output", "results"])
        context_entry = self._build_context_entry(current_step, eval_result, history_fields)
        
        new_history = state["context_history"] + [context_entry]
        
        # Extract goal for display purposes (use gold alternative)
        reference_alt = self._get_gold_alternative(current_step)
        # Handle multi-step alternatives (list) vs single step (dict)
        if isinstance(reference_alt, list):
            goal = reference_alt[0].get("goal", "") if reference_alt else ""
        else:
            goal = reference_alt.get("goal", "")
        
        # Show result status with color coding
        if eval_result and eval_result.get("completed"):
            status = f"{Colors.GREEN}✓ COMPLETED{Colors.RESET}"
        else:
            status = f"{Colors.RED}✗ NOT COMPLETED{Colors.RESET}"
        
        print(f"{status} Step {step_index + 1} - Goal: {goal}\n")
        
        return {
            "current_step_index": step_index + 1,
            "context_history": new_history,
            "completed_results": new_completed_results,
            "_step_eval_result": None,  # Clear the temp result
            "_accumulated_commands": None,  # Clear accumulated commands
            "_is_fine_grained": False  # Reset fine-grained flag
        }
    
    def _get_gold_alternative(self, step_data: Dict[str, Any]) -> Any:
        """Extract gold standard alternative from step data.
        
        In the step data structure, each step with alternatives has exactly one marked
        as gold: true and zero or more marked as gold: false. This method finds and
        returns the gold standard alternative.
        
        Args:
            step_data: The step data which may contain "or" alternatives
            
        Returns:
            - For single steps (no "or"): returns the step itself (dict)
            - For atomic gold alternatives: returns the dict
            - For multi-step gold alternatives: returns the FULL list of steps
            Falls back to first alternative if no gold marker found (malformed data).
        """
        if "or" not in step_data:
            return step_data
        
        # Search for the alternative marked as gold: true
        for alt in step_data["or"]:
            if isinstance(alt, list):
                # Multi-step alternative - check if first step is marked as gold
                if alt and alt[0].get("gold", False):
                    return alt  # Return the full list, not just alt[0]
            else:
                # Atomic alternative - check if marked as gold
                if alt.get("gold", False):
                    return alt
        
        # Fallback to first alternative if no gold marker found (indicates malformed data)
        # In well-formed step data, there should always be exactly one gold: true
        print(f"{Colors.YELLOW}⚠ Warning: No gold alternative found in step. Using first alternative as fallback.{Colors.RESET}")
        return step_data["or"][0]  # Return as-is (could be list or dict)
    
    def _build_context_entry(
        self,
        current_step: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]],
        history_fields: List[str]
    ) -> str:
        """Build a context history entry based on configured fields.
        
        Args:
            current_step: The step definition from steps_enriched.json
            eval_result: The evaluation result (contains matched_command)
            history_fields: List of fields to include ("goal", "command", "output", "results")
            
        Returns:
            Formatted context entry string
        """
        # Extract reference alternative for gold standard data
        reference_alt = self._get_gold_alternative(current_step)
        
        # Check if it's a multi-step alternative (list) or single step (dict)
        if isinstance(reference_alt, list):
            # Multi-step alternative - build entry for each sub-step
            all_parts = []
            for i, sub_step in enumerate(reference_alt, 1):
                sub_parts = []
                if len(reference_alt) > 1:
                    sub_parts.append(f"Sub-step {i}:")
                
                if "goal" in history_fields:
                    goal = sub_step.get("goal", "")
                    if goal:
                        sub_parts.append(f"  Goal: {goal}")
                
                if "command" in history_fields:
                    command = sub_step.get("command", "")
                    if command:
                        sub_parts.append(f"  Command: {command}")
                
                if "output" in history_fields:
                    output = sub_step.get("output", "")
                    if output:
                        truncated_output = output[:500] + "..." if len(output) > 500 else output
                        sub_parts.append(f"  Output: {truncated_output}")
                
                if "results" in history_fields:
                    results = sub_step.get("results", [])
                    if results:
                        results_str = "; ".join(results)
                        sub_parts.append(f"  Results: {results_str}")
                
                all_parts.extend(sub_parts)
            
            return "\n".join(all_parts) if all_parts else "Completed"
        
        else:
            # Single step (atomic alternative or no alternatives)
            parts = []
            
            if "goal" in history_fields:
                goal = reference_alt.get("goal", "")
                if goal:
                    parts.append(f"Goal: {goal}")
            
            if "command" in history_fields:
                # Use matched command from eval_result if available, otherwise gold command
                if eval_result and eval_result.get("matched_command"):
                    command = eval_result["matched_command"]
                else:
                    command = reference_alt.get("command", "")
                if command:
                    parts.append(f"Command: {command}")
            
            if "output" in history_fields:
                output = reference_alt.get("output", "")
                if output:
                    # Truncate output if too long (keep first 500 chars)
                    truncated_output = output[:500] + "..." if len(output) > 500 else output
                    parts.append(f"Output: {truncated_output}")
            
            if "results" in history_fields:
                results = reference_alt.get("results", [])
                if results:
                    results_str = "; ".join(results)
                    parts.append(f"Results: {results_str}")
            
            # Join all parts with newlines for readability
            return "\n".join(parts) if parts else "Completed"
    
    def _finalize_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Finalize evaluation and format results.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with formatted results
        """
        print(f"\n{Colors.BOLD}=== Finalizing evaluation ==={Colors.RESET}\n")
        
        results = format_evaluation_results(state)
        
        # Count example steps (excluded from scoring)
        example_steps = sum(1 for r in state['completed_results'] if r.get('_example_step', False))
        total_evaluated_steps = len(state['steps']) - example_steps
        completed_steps = sum(1 for r in state['completed_results'] 
                             if self._is_step_completed(r) and not r.get('_example_step', False))
        
        print(f"Final score: {results['score']:.2%}")
        if example_steps > 0:
            print(f"{Colors.BOLD}Completed {completed_steps} / {total_evaluated_steps} steps ({example_steps} example step(s) excluded){Colors.RESET}\n")
        else:
            print(f"{Colors.BOLD}Completed {completed_steps} / {total_evaluated_steps} steps{Colors.RESET}\n")
        
        return {"completed_results": state["completed_results"]}
    
    def _is_step_completed(self, step_result: Any) -> bool:
        """Check if a step result indicates completion.
        
        Args:
            step_result: Step result to check
            
        Returns:
            True if step was completed
        """
        if isinstance(step_result, dict):
            if "or" in step_result:
                # Check if any alternative was completed
                for alt in step_result["or"]:
                    if isinstance(alt, list):
                        if any(sub.get("completed", False) for sub in alt):
                            return True
                    elif alt.get("completed", False):
                        return True
            elif step_result.get("completed", False):
                return True
        return False
    
    # Step subgraph nodes
    
    def _prompt_agent_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Prompt the agent under test for the next command.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with agent's prediction
        """
        iteration = state["current_iteration"]
        print(f"  Iteration {iteration + 1}/{state['max_iterations_per_step']}: Prompting agent...")
        
        # Build context for this step
        context = build_step_context(state, state["current_step_index"])
        
        # Get agent's prediction with Phoenix tracing
        try:
            # If Phoenix is enabled, create a span for the agent prompt
            if self.enable_phoenix and trace:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(
                    "prompt_agent",
                    attributes={
                        "step_index": state["current_step_index"],
                        "iteration": iteration + 1,
                        "max_iterations": state["max_iterations_per_step"],
                        "task_mode": state.get("task_mode", "command"),
                        "evaluation_protocol": state.get("evaluation_protocol", "match_alternatives"),
                        "challenge": state["challenge_name"],
                        # Add prompt directly as attributes for better visibility
                        "input": context,
                        "input.value": context,  # Phoenix looks for this
                        "input.mime_type": "text/plain"
                    }
                ) as span:
                    # Also log as event for historical record
                    span.add_event(
                        "sending_prompt_to_agent",
                        attributes={
                            "prompt": context,
                            "prompt_length": len(context),
                            "agent_type": "a2a" if hasattr(self.agent_interface, 'agent_url') else "internal"
                        }
                    )
                    
                    prediction = self.agent_interface.predict_next_step(context)
                    
                    # Log the response with Phoenix-friendly attributes
                    span.set_attribute("output", prediction)
                    span.set_attribute("output.value", prediction)  # Phoenix looks for this
                    span.set_attribute("output.mime_type", "text/plain")
                    
                    span.add_event(
                        "received_agent_response",
                        attributes={
                            "prediction": prediction,
                            "prediction_length": len(prediction)
                        }
                    )
            else:
                # No Phoenix tracing
                prediction = self.agent_interface.predict_next_step(context)
            
            print(f"  Agent predicted: {prediction}")
            
            # Add to predictions list
            new_predictions = state["agent_predictions"] + [prediction]
            
            return {
                "agent_predictions": new_predictions,
                "current_iteration": iteration + 1
            }
        except Exception as e:
            print(f"  Error getting agent prediction: {e}")
            if self.enable_phoenix and trace:
                # Log the error to Phoenix
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span("prompt_agent_error") as span:
                    span.record_exception(e)
                    span.set_attribute("error", str(e))
            return {
                "current_iteration": iteration + 1,
                "current_step_goal_reached": True  # Stop on error
            }
    
    def _evaluate_response_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Evaluate the agent's response against expected alternatives.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with evaluation result
        """
        if not state["agent_predictions"]:
            return {"current_step_goal_reached": True}
        
        step_index = state["current_step_index"]
        current_step = state["steps"][step_index]
        
        # Build accumulated commands if we're in fine-grained mode
        accumulated = state.get("_accumulated_commands")
        if accumulated:
            # Evaluate the accumulated sequence
            commands_to_evaluate = accumulated
        else:
            # Evaluate just the last prediction
            commands_to_evaluate = state["agent_predictions"][-1]
        
        # Extract alternatives and goal from step
        if "or" in current_step:
            alternatives = current_step["or"]
            reference_alt = alternatives[0]
            if isinstance(reference_alt, list):
                reference_alt = reference_alt[0]
        else:
            alternatives = [current_step]
            reference_alt = current_step
        
        goal = reference_alt.get("goal", "")
        
        print(f"  Evaluating {'accumulated commands' if accumulated else 'prediction'} against {len(alternatives)} alternatives...")
        if accumulated:
            # Format multi-line commands for logging
            cmd_list = accumulated.split("\n")
            if len(cmd_list) == 1:
                print(f"  Accumulated commands: {accumulated}")
            else:
                print(f"  Accumulated commands ({len(cmd_list)} commands):")
                for i, cmd in enumerate(cmd_list, 1):
                    print(f"    {i}. {cmd}")
        
        # Evaluate the prediction(s)
        try:
            eval_result = self.step_evaluator.evaluate_prediction(
                commands_to_evaluate,
                alternatives,
                goal
            )
            
            # Color code the match result
            match_color = Colors.GREEN if eval_result['completed'] else Colors.RED
            print(f"  {match_color}Match: {eval_result['completed']}, "
                  f"Confidence: {eval_result['confidence']:.2f}, "
                  f"Alternative: {eval_result['matched_alternative_index']}{Colors.RESET}")
            
            # Check if goal is reached
            goal_check = self.step_evaluator.check_goal_reached(
                eval_result,
                current_step,
                state["current_iteration"],
                state["max_iterations_per_step"],
                accumulated_commands=accumulated,
                include_goal=state.get("include_goal", "always")
            )
            
            # Color code the goal check based on whether goal was reached
            goal_color = Colors.GREEN if goal_check['goal_reached'] else Colors.YELLOW
            print(f"  {goal_color}Goal check: {goal_check['reason']}{Colors.RESET}")
            
            # Handle fine-grained commands - accumulate them
            # But only if we haven't hit max iterations
            if goal_check.get("is_fine_grained", False) and state["current_iteration"] < state["max_iterations_per_step"]:
                if not accumulated:
                    # Start accumulating commands (keep as newline-separated, not one-liner)
                    new_accumulated = "\n".join(state["agent_predictions"])
                    return {
                        "current_step_goal_reached": False,
                        "_is_fine_grained": True,
                        "_accumulated_commands": new_accumulated,
                        "_step_eval_result": None
                    }
                else:
                    # Continue accumulating - add latest command on new line
                    new_accumulated = accumulated + "\n" + state["agent_predictions"][-1]
                    return {
                        "current_step_goal_reached": False,
                        "_is_fine_grained": True,
                        "_accumulated_commands": new_accumulated,
                        "_step_eval_result": None
                    }
            
            # Store evaluation result for use in record_result_node
            # Use accumulated commands if available, otherwise last prediction
            agent_prediction = accumulated if accumulated else state["agent_predictions"][-1]
            # Always store the agent's prediction (for both matched and unmatched cases)
            eval_result["agent_prediction"] = agent_prediction
            if eval_result["completed"]:
                eval_result["matched_command"] = agent_prediction
            
            return {
                "current_step_goal_reached": goal_check["goal_reached"],
                "_step_eval_result": eval_result,  # Always pass eval_result (not just on match)
                "_is_fine_grained": False,
                "_accumulated_commands": None
            }
            
        except Exception as e:
            print(f"  {Colors.RED}Error during evaluation: {e}{Colors.RESET}")
            return {"current_step_goal_reached": True}
    
    def _build_step_result(self, step_data: Any, eval_result: Optional[Dict[str, Any]], task_mode: str = "command") -> Dict[str, Any]:
        """Build the result structure for a completed step.
        
        Args:
            step_data: The step data from steps_enriched.json
            eval_result: Evaluation result (None if no match found)
            task_mode: The task mode (command, goal, or anticipated_result)
            
        Returns:
            Step result structure matching the output format
        """
        # Determine field names based on task mode
        if task_mode == "goal":
            original_field = "original_goal"
            matched_field = "matched_prediction"
            unmatched_field = "unmatched_prediction"
            source_field = "goal"
        elif task_mode == "anticipated_result":
            original_field = "original_anticipated_result"
            matched_field = "matched_prediction"
            unmatched_field = "unmatched_prediction"
            source_field = "results"  # Use "results" field from steps_enriched.json
        else:  # command
            original_field = "original_command"
            matched_field = "matched_prediction"
            unmatched_field = "unmatched_prediction"
            source_field = "command"
        
        # Extract agent's prediction if available
        agent_prediction = eval_result.get("agent_prediction") if eval_result else None
        
        if "or" in step_data:
            # Step with alternatives
            or_results = []
            for i, alternative in enumerate(step_data["or"]):
                if isinstance(alternative, list):
                    # Multi-step alternative
                    sub_results = []
                    for sub_step in alternative:
                        sub_result = {
                            "completed": False,
                            original_field: sub_step.get(source_field, ""),
                            "gold": sub_step.get("gold", False)
                        }
                        # Mark as completed if this alternative matched
                        if eval_result and eval_result["matched_alternative_index"] == i:
                            sub_result["completed"] = True
                            if agent_prediction:
                                sub_result[matched_field] = agent_prediction
                        else:
                            # Add unmatched prediction for non-matching alternatives
                            if agent_prediction:
                                sub_result[unmatched_field] = agent_prediction
                        sub_results.append(sub_result)
                    or_results.append(sub_results)
                else:
                    # Atomic alternative
                    alt_result = {
                        "completed": False,
                        original_field: alternative.get(source_field, ""),
                        "gold": alternative.get("gold", False)
                    }
                    # Mark as completed if this alternative matched
                    if eval_result and eval_result["matched_alternative_index"] == i:
                        alt_result["completed"] = True
                        if agent_prediction:
                            alt_result[matched_field] = agent_prediction
                    else:
                        # Add unmatched prediction for non-matching alternatives
                        if agent_prediction:
                            alt_result[unmatched_field] = agent_prediction
                    or_results.append(alt_result)
            return {"or": or_results}
        else:
            # Single step without alternatives
            is_completed = eval_result and eval_result.get("completed", False)
            result = {
                "completed": is_completed,
                original_field: step_data.get(source_field, ""),
                "gold": step_data.get("gold", False)
            }
            # Add prediction with appropriate field name
            if agent_prediction:
                if is_completed:
                    result[matched_field] = agent_prediction
                else:
                    result[unmatched_field] = agent_prediction
            return result
    
    # Conditional edge functions
    
    def _check_more_steps(self, state: EvaluationState) -> Literal["continue", "done"]:
        """Check if there are more steps to evaluate.
        
        Called after record_result to decide if we should continue to the next step.
        
        Args:
            state: Current state
            
        Returns:
            "continue" if more steps remain, "done" if all steps completed
        """
        if state["current_step_index"] < len(state["steps"]):
            return "continue"
        return "done"
    
    def _check_goal_reached(self, state: EvaluationState) -> Literal["continue", "done"]:
        """Check if the current step's goal has been reached.
        
        Args:
            state: Current state
            
        Returns:
            Next node to visit in subgraph
        """
        # Safety check: stop if we've exceeded max iterations
        if state["current_iteration"] >= state["max_iterations_per_step"]:
            print(f"  {Colors.YELLOW}⚠ Max iterations ({state['max_iterations_per_step']}) reached, stopping{Colors.RESET}")
            return "done"
        
        if state["current_step_goal_reached"]:
            return "done"
        return "continue"
    
    def run(
        self,
        challenge_name: str,
        agent_llm_config: Dict[str, Any],
        evaluator_llm_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the evaluation workflow.
        
        Args:
            challenge_name: Name of the CTF challenge
            agent_llm_config: LLM configuration for agent
            evaluator_llm_config: LLM configuration for evaluator
            
        Returns:
            Final evaluation results
        """
        initial_state: EvaluationState = {
            "challenge_name": challenge_name,
            "steps": [],
            "current_step_index": 0,
            "completed_results": [],
            "agent_predictions": [],
            "context_history": [],
            "agent_llm_config": agent_llm_config,
            "evaluator_llm_config": evaluator_llm_config,
            "max_iterations_per_step": self.max_iterations_per_step,
            "include_goal": self.include_goal,
            "include_tactic": self.include_tactic,
            "include_prerequisites": self.include_prerequisites,
            "history_context": self.history_context,
            "evaluation_protocol": self.step_evaluator.evaluation_protocol,
            "task_mode": self.step_evaluator.task_mode,
            "current_iteration": 0,
            "current_step_goal_reached": False,
            "_step_eval_result": None,
            "_is_fine_grained": False,
            "_accumulated_commands": None
        }
        
        # Run the workflow with recursion limit
        # Account for: load_challenge + (prepare_step + evaluate_step + record_result) * num_steps + finalize
        # Using a generous limit to handle challenges with many steps
        # Increased to 500 to handle subgraph nesting
        config = {
            "recursion_limit": 500,  # Generous limit for nested subgraphs and many steps
            "configurable": {"thread_id": challenge_name}
        }
        final_state = self.graph.invoke(initial_state, config)
        
        # Format and return results
        return format_evaluation_results(final_state)
    
    def run_batch(
        self,
        challenge_names: List[str],
        agent_llm_config: Dict[str, Any],
        evaluator_llm_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run evaluation workflow on multiple challenges.
        
        Args:
            challenge_names: List of CTF challenge names
            agent_llm_config: LLM configuration for agent
            evaluator_llm_config: LLM configuration for evaluator
            
        Returns:
            Aggregated evaluation results for all challenges
        """
        from .utils import format_batch_results
        
        all_results = []
        
        for challenge_name in challenge_names:
            print(f"\n{'=' * 70}")
            print(f"Evaluating challenge: {challenge_name}")
            print('=' * 70)
            
            try:
                result = self.run(
                    challenge_name=challenge_name,
                    agent_llm_config=agent_llm_config,
                    evaluator_llm_config=evaluator_llm_config
                )
                all_results.append(result)
                print(f"{Colors.GREEN}✓ Completed {challenge_name}: Score = {result['score']:.2%}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}✗ Failed {challenge_name}: {e}{Colors.RESET}")
                error_result = {
                    "challenge": challenge_name,
                    "error": str(e),
                    "score": 0.0,
                    "steps_completed": []
                }
                all_results.append(error_result)
        
        return format_batch_results(all_results)

