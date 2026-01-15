"""State definitions for the evaluator workflow."""

from typing import TypedDict, List, Dict, Any, Optional


class EvaluationState(TypedDict, total=False):
    """State schema for the evaluation workflow.
    
    This TypedDict defines all fields that can be present in the workflow state.
    total=False allows fields to be optional.
    """
    # Challenge information
    challenge_name: str
    steps: List[Any]
    
    # Step tracking
    current_step_index: int
    completed_results: List[Dict[str, Any]]
    
    # Agent interaction
    agent_predictions: List[str]
    context_history: List[str]
    
    # Configuration
    agent_llm_config: Dict[str, Any]
    evaluator_llm_config: Dict[str, Any]
    max_iterations_per_step: int
    include_goal: str  # "first", "always", or "never"
    include_tactic: str  # "first", "always", or "never"
    include_prerequisites: str  # "first", "always", or "never"
    history_context: List[str]  # Fields to include in context history: "goal", "command", "output", "results"
    evaluation_protocol: str  # "match_alternatives" or "single_path"
    task_mode: str  # "command", "anticipated_result", or "goal"
    
    # Current iteration state
    current_iteration: int
    current_step_goal_reached: bool
    
    # Temporary/internal state (prefixed with _)
    _step_eval_result: Optional[Dict[str, Any]]
    _is_fine_grained: bool
    _accumulated_commands: Optional[str]
