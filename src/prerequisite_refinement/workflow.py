"""LangGraph workflow for refining step prerequisites by analyzing information gaps."""

import os
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from ..evaluator.state import EvaluationState
from ..evaluator.utils import build_step_context, load_challenge_steps
from .analyzer import PrerequisiteAnalyzer

# Phoenix imports (optional - only if available)
try:
    from phoenix.otel import register
    from opentelemetry import trace
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    register = None
    trace = None


class PrerequisiteRefinementWorkflow:
    """LangGraph workflow for analyzing and refining step prerequisites."""
    
    def __init__(
        self,
        prerequisite_analyzer: PrerequisiteAnalyzer,
        enable_phoenix: bool = True
    ):
        """Initialize the prerequisite refinement workflow.
        
        Args:
            prerequisite_analyzer: Analyzer for identifying missing prerequisites
            enable_phoenix: Whether to enable Phoenix tracing (default: True)
        """
        self.prerequisite_analyzer = prerequisite_analyzer
        self.enable_phoenix = enable_phoenix and PHOENIX_AVAILABLE
        
        # Check if Phoenix is already initialized
        self.phoenix_tracer = None
        if self.enable_phoenix:
            try:
                if trace and trace.get_tracer_provider().__class__.__name__ != 'ProxyTracerProvider':
                    print("✓ Phoenix tracing active (initialized before workflow creation)")
                    self.phoenix_tracer = trace.get_tracer_provider()
                else:
                    print("⚠ Phoenix not initialized early - initializing now")
                    self._init_phoenix()
            except Exception as e:
                print(f"⚠ Warning: Phoenix check failed: {e}")
                self.enable_phoenix = False
                self.phoenix_tracer = None
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _init_phoenix(self) -> None:
        """Initialize Phoenix for observability and tracing (fallback method)."""
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            
            project_name = os.getenv("PHOENIX_PROJECT_NAME", "bracegreen-prerequisite-refinement")
            
            # Register Phoenix tracer with selective instrumentation to avoid double logging
            self.phoenix_tracer = register(
                project_name=project_name,
                auto_instrument=False  # Manual instrumentation to avoid double logging
            )
            
            # Manually instrument only LangChain (which includes ChatLiteLLM)
            LangChainInstrumentor().instrument(tracer_provider=self.phoenix_tracer)
            
            print("✓ Phoenix tracing enabled (fallback initialization)")
            print(f"  Project: {project_name}")
            print(f"  Instrumentation: LangChain only (no double logging)")
            
        except Exception as e:
            print(f"⚠ Warning: Could not initialize Phoenix tracing: {e}")
            print("  Continuing without Phoenix tracing...")
            self.enable_phoenix = False
            self.phoenix_tracer = None
    
    def _build_graph(self) -> StateGraph:
        """Build the prerequisite refinement workflow graph.
        
        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(EvaluationState)
        
        # Add nodes
        workflow.add_node("load_challenge", self._load_challenge_node)
        workflow.add_node("prepare_step", self._prepare_step_node)
        workflow.add_node("analyze_prerequisites", self._analyze_prerequisites_node)
        workflow.add_node("record_analysis", self._record_analysis_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Add edges
        workflow.add_edge(START, "load_challenge")
        workflow.add_edge("load_challenge", "prepare_step")
        workflow.add_edge("prepare_step", "analyze_prerequisites")
        workflow.add_edge("analyze_prerequisites", "record_analysis")
        workflow.add_conditional_edges(
            "record_analysis",
            self._check_more_steps,
            {
                "continue": "prepare_step",
                "done": "finalize"
            }
        )
        workflow.add_edge("finalize", END)
        
        # Compile with memory for checkpointing
        return workflow.compile(checkpointer=MemorySaver())
    
    # Graph nodes
    
    def _load_challenge_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Load challenge steps and initialize state.
        
        Args:
            state: Current state
            
        Returns:
            Updated state fields
        """
        print(f"\n=== Loading challenge for prerequisite analysis: {state['challenge_name']} ===\n")
        
        # Steps may already be loaded in run() method to calculate recursion limit
        # Only load if not already present
        if not state.get("steps") or len(state["steps"]) == 0:
            steps = load_challenge_steps(state["challenge_name"])
        else:
            steps = state["steps"]
        
        print(f"Challenge has {len(steps)} steps to analyze\n")
        
        return {
            "steps": steps,
            "completed_results": [],  # Will store analysis results instead
            "current_step_index": 0,
            "context_history": []
        }
    
    def _prepare_step_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Prepare for analyzing the next step.
        
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
            
            print(f"\n=== Analyzing step {step_index + 1}/{len(state['steps'])} ===")
            print(f"Goal: {goal}")
            if tactic:
                print(f"Tactic: {tactic}")
            print()
        
        return {}
    
    def _analyze_prerequisites_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Analyze the current step to identify missing prerequisites.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with analysis result
        """
        step_index = state["current_step_index"]
        current_step = state["steps"][step_index]
        
        # Build the context that would be provided to an agent
        # Note: We build context with "gold standard" history (teacher forcing)
        context = build_step_context(state, step_index)
        
        # Get previous steps data for reference
        previous_steps_data = state["steps"][:step_index] if step_index > 0 else []
        
        print(f"  Analyzing information gaps for step {step_index + 1}...")
        
        try:
            analysis_result = self.prerequisite_analyzer.analyze_step(
                step_context=context,
                step_data=current_step,
                step_index=step_index,
                previous_steps_data=previous_steps_data
            )
            
            # Display results
            print(f"\n  ✓ Analysis complete:")
            print(f"    Current prerequisites: {len(analysis_result['current_prerequisites'])}")
            print(f"    Missing information items: {len(analysis_result['missing_information'])}")
            print(f"    Suggested prerequisites: {len(analysis_result['suggested_prerequisites'])}")
            
            if analysis_result['missing_information']:
                print(f"\n  Missing information identified:")
                for item in analysis_result['missing_information']:
                    print(f"    - [{item['criticality'].upper()}] {item['item']}")
                    print(f"      Source: {item['source']}")
            
            return {"_step_analysis_result": analysis_result}
            
        except Exception as e:
            print(f"  ✗ Error during analysis: {e}")
            return {"_step_analysis_result": None}
    
    def _record_analysis_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Record the analysis result for the current step.
        
        Args:
            state: Current state
            
        Returns:
            Updated state fields
        """
        step_index = state["current_step_index"]
        current_step = state["steps"][step_index]
        analysis_result = state.get("_step_analysis_result")
        
        # Extract goal from step
        if "or" in current_step:
            reference_alt = current_step["or"][0]
            if isinstance(reference_alt, list):
                reference_alt = reference_alt[0]
        else:
            reference_alt = current_step
        
        goal = reference_alt.get("goal", "")
        
        # Build result structure
        step_result = {
            "step_index": step_index,
            "goal": goal,
            "tactic": reference_alt.get("tactic", ""),
            "analysis": analysis_result if analysis_result else {
                "error": "Analysis failed"
            }
        }
        
        # Add to completed results
        new_completed_results = state["completed_results"] + [step_result]
        
        # Add to context history (for building context for future steps)
        context_entry = f"Completed: {goal}"
        new_history = state["context_history"] + [context_entry]
        
        # Show summary
        if analysis_result and analysis_result.get("missing_information"):
            status = f"⚠ {len(analysis_result['missing_information'])} gaps identified"
        else:
            status = "✓ No critical gaps"
        
        print(f"\n{status} - Step {step_index + 1}: {goal}\n")
        
        return {
            "current_step_index": step_index + 1,
            "context_history": new_history,
            "completed_results": new_completed_results,
            "_step_analysis_result": None  # Clear the temp result
        }
    
    def _finalize_node(self, state: EvaluationState) -> Dict[str, Any]:
        """Finalize analysis and format results.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with formatted results
        """
        print("\n=== Finalizing prerequisite analysis ===\n")
        
        total_steps = len(state["steps"])
        total_gaps = sum(
            len(result.get("analysis", {}).get("missing_information", []))
            for result in state["completed_results"]
        )
        
        print(f"Analyzed {total_steps} steps")
        print(f"Total information gaps identified: {total_gaps}\n")
        
        return {"completed_results": state["completed_results"]}
    
    # Conditional edge functions
    
    def _check_more_steps(self, state: EvaluationState) -> Literal["continue", "done"]:
        """Check if there are more steps to analyze.
        
        Args:
            state: Current state
            
        Returns:
            "continue" if more steps remain, "done" if all steps analyzed
        """
        if state["current_step_index"] < len(state["steps"]):
            return "continue"
        return "done"
    
    def run(
        self,
        challenge_name: str,
        analyzer_llm_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the prerequisite refinement workflow.
        
        Args:
            challenge_name: Name of the CTF challenge
            analyzer_llm_config: LLM configuration for analyzer
            
        Returns:
            Analysis results for all steps
        """
        # Load steps first to calculate recursion limit
        steps = load_challenge_steps(challenge_name)
        
        # Calculate recursion limit: steps * 5 + buffer for safety
        # Each step goes through: prepare_step -> analyze_prerequisites -> record_analysis
        # With conditional edges, we need room for all step iterations
        recursion_limit = len(steps) * 5 + 10
        
        print(f"Setting recursion limit: {recursion_limit} (for {len(steps)} steps)")
        
        initial_state: EvaluationState = {
            "challenge_name": challenge_name,
            "steps": steps,  # Pre-load steps to avoid reloading
            "current_step_index": 0,
            "completed_results": [],
            "agent_predictions": [],
            "context_history": [],
            "agent_llm_config": {},  # Not used in this workflow
            "evaluator_llm_config": analyzer_llm_config,
            "max_iterations_per_step": 1,  # Not used in this workflow
            "current_iteration": 0,
            "current_step_goal_reached": False,
            "_step_eval_result": None,
            "_is_fine_grained": False,
            "_accumulated_commands": None,
            "_step_analysis_result": None
        }
        
        # Run the workflow with appropriate recursion limit
        config = {
            "configurable": {
                "thread_id": f"{challenge_name}_prereq_analysis"
            },
            "recursion_limit": recursion_limit
        }
        
        print(f"Invoking workflow with recursion_limit={recursion_limit}\n")
        final_state = self.graph.invoke(initial_state, config)
        
        # Return analysis results
        return {
            "challenge": challenge_name,
            "total_steps": len(final_state["steps"]),
            "analyses": final_state["completed_results"]
        }

