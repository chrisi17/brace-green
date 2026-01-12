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


class EvaluatorWorkflow:
    """LangGraph workflow for step-by-step agent evaluation."""
    
    def __init__(
        self,
        agent_interface: LLMAgentInterface,
        step_evaluator: StepEvaluator,
        max_iterations_per_step: int = 10,
        enable_phoenix: bool = True
    ):
        """Initialize the evaluator workflow.
        
        Args:
            agent_interface: Interface to the agent under test
            step_evaluator: Evaluator for comparing predictions
            max_iterations_per_step: Maximum iterations per step
            enable_phoenix: Whether to enable Phoenix tracing (default: True)
        """
        self.agent_interface = agent_interface
        self.step_evaluator = step_evaluator
        self.max_iterations_per_step = max_iterations_per_step
        self.enable_phoenix = enable_phoenix and PHOENIX_AVAILABLE
        
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
                    print("⚠ Phoenix not initialized early - initializing now (may affect trace context)")
                    self._init_phoenix()
            except Exception as e:
                print(f"⚠ Warning: Phoenix check failed: {e}")
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
            print(f"⚠ Warning: Could not initialize Phoenix tracing: {e}")
            print("  Continuing without Phoenix tracing...")
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
        
        # Extract goal for logging
        if step_index < len(state["steps"]):
            current_step = state["steps"][step_index]
            if "or" in current_step:
                reference_alt = current_step["or"][0]
                if isinstance(reference_alt, list):
                    reference_alt = reference_alt[0]
            else:
                reference_alt = current_step
            goal = reference_alt.get("goal", "Unknown goal")
            tactic = reference_alt.get("tactic", "")
            
            print(f"\n=== Preparing step {step_index + 1}/{len(state['steps'])} ===")
            print(f"Goal: {goal}")
            if tactic:
                print(f"Tactic: {tactic}")
            print()
        else:
            print(f"\n=== Preparing step {step_index + 1}/{len(state['steps'])} ===\n")
        
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
        
        # Extract goal for logging
        if "or" in current_step:
            reference_alt = current_step["or"][0]
            if isinstance(reference_alt, list):
                reference_alt = reference_alt[0]
        else:
            reference_alt = current_step
        goal = reference_alt.get("goal", "Unknown goal")
        
        print(f"=== Evaluating step {step_index + 1} ===")
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
        step_result = self._build_step_result(current_step, eval_result)
        
        # Add to completed results
        new_completed_results = state["completed_results"] + [step_result]
        
        # Add to context history for future steps (gold standard path)
        if "or" in current_step:
            reference_alt = current_step["or"][0]
            if isinstance(reference_alt, list):
                reference_alt = reference_alt[0]
        else:
            reference_alt = current_step
        
        goal = reference_alt.get("goal", "")
        context_entry = f"Completed: {goal}"
        
        new_history = state["context_history"] + [context_entry]
        
        # Show result status
        if eval_result and eval_result.get("completed"):
            status = "✓ COMPLETED"
        else:
            status = "✗ NOT COMPLETED"
        
        print(f"{status} Step {step_index + 1} - Goal: {goal}\n")
        
        return {
            "current_step_index": step_index + 1,
            "context_history": new_history,
            "completed_results": new_completed_results,
            "_step_eval_result": None,  # Clear the temp result
            "_accumulated_commands": None,  # Clear accumulated commands
            "_is_fine_grained": False  # Reset fine-grained flag
        }
    
    def _finalize_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Finalize evaluation and format results.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with formatted results
        """
        print("\n=== Finalizing evaluation ===\n")
        
        results = format_evaluation_results(state)
        
        print(f"Final score: {results['score']:.2%}")
        print(f"Completed {sum(1 for r in state['completed_results'] if self._is_step_completed(r))} / {len(state['steps'])} steps\n")
        
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
        
        # Get agent's prediction
        try:
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
            
            print(f"  Match: {eval_result['completed']}, "
                  f"Confidence: {eval_result['confidence']:.2f}, "
                  f"Alternative: {eval_result['matched_alternative_index']}")
            
            # Check if goal is reached
            goal_check = self.step_evaluator.check_goal_reached(
                eval_result,
                current_step,
                state["current_iteration"],
                state["max_iterations_per_step"],
                accumulated_commands=accumulated
            )
            
            print(f"  Goal check: {goal_check['reason']}")
            
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
            matched_command = accumulated if accumulated else state["agent_predictions"][-1]
            if eval_result["completed"]:
                eval_result["matched_command"] = matched_command
            
            return {
                "current_step_goal_reached": goal_check["goal_reached"],
                "_step_eval_result": eval_result if eval_result["completed"] else None,
                "_is_fine_grained": False,
                "_accumulated_commands": None
            }
            
        except Exception as e:
            print(f"  Error during evaluation: {e}")
            return {"current_step_goal_reached": True}
    
    def _build_step_result(self, step_data: Any, eval_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the result structure for a completed step.
        
        Args:
            step_data: The step data from steps_enriched.json
            eval_result: Evaluation result (None if no match found)
            
        Returns:
            Step result structure matching the output format
        """
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
                            "original_command": sub_step.get("command", ""),
                            "gold": sub_step.get("gold", False)
                        }
                        # Mark as completed if this alternative matched
                        if eval_result and eval_result["matched_alternative_index"] == i:
                            sub_result["completed"] = True
                            sub_result["matched_command"] = eval_result["matched_command"]
                        sub_results.append(sub_result)
                    or_results.append(sub_results)
                else:
                    # Atomic alternative
                    alt_result = {
                        "completed": False,
                        "original_command": alternative.get("command", ""),
                        "gold": alternative.get("gold", False)
                    }
                    # Mark as completed if this alternative matched
                    if eval_result and eval_result["matched_alternative_index"] == i:
                        alt_result["completed"] = True
                        alt_result["matched_command"] = eval_result["matched_command"]
                    or_results.append(alt_result)
            return {"or": or_results}
        else:
            # Single step without alternatives
            result = {
                "completed": eval_result is not None,
                "original_command": step_data.get("command", ""),
                "gold": step_data.get("gold", False)
            }
            if eval_result:
                result["matched_command"] = eval_result["matched_command"]
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
            print(f"  ⚠ Max iterations ({state['max_iterations_per_step']}) reached, stopping")
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
                print(f"✓ Completed {challenge_name}: Score = {result['score']:.2%}")
            except Exception as e:
                print(f"✗ Failed {challenge_name}: {e}")
                error_result = {
                    "challenge": challenge_name,
                    "error": str(e),
                    "score": 0.0,
                    "steps_completed": []
                }
                all_results.append(error_result)
        
        return format_batch_results(all_results)

