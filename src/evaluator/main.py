"""CLI entry point for the evaluator workflow."""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from .agent_interface import LLMAgentInterface, create_agent_interface
from .step_evaluator import StepEvaluator
from .workflow import EvaluatorWorkflow
from .utils import save_evaluation_results, discover_all_challenges


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a red-teaming agent on a CTF challenge using LangGraph workflow"
    )
    
    # Challenge specification (required unless in serve mode)
    parser.add_argument(
        "--challenge",
        type=str,
        default=None,
        help="Name of the CTF challenge (e.g., 'Funbox')"
    )
    parser.add_argument(
        "--challenges",
        type=str,
        default=None,
        help="Comma-separated list of challenges, or 'all' for all challenges"
    )
    
    # Agent mode selection
    agent_mode = parser.add_mutually_exclusive_group()
    agent_mode.add_argument(
        "--internal",
        action="store_true",
        help="Use internal LLM agent (default mode)"
    )
    agent_mode.add_argument(
        "--agent-url",
        type=str,
        default=None,
        help="URL of remote A2A agent to evaluate"
    )
    
    # Agent configuration (for internal mode)
    parser.add_argument(
        "--agent-model",
        type=str,
        default="gpt-4o",
        help="Model for the agent under test (default: gpt-4o) [internal mode only]"
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=0.7,
        help="Temperature for agent LLM (default: 0.7) [internal mode only]"
    )
    parser.add_argument(
        "--agent-max-tokens",
        type=int,
        default=500,
        help="Max tokens for agent LLM (default: 500) [internal mode only]"
    )
    parser.add_argument(
        "--agent-timeout",
        type=float,
        default=300.0,
        help="Timeout for A2A agent calls in seconds (default: 300) [a2a mode only]"
    )
    
    # Evaluator configuration
    parser.add_argument(
        "--evaluator-model",
        type=str,
        default="gpt-4o",
        help="Model for the evaluator (default: gpt-4o)"
    )
    parser.add_argument(
        "--evaluator-max-tokens",
        type=int,
        default=2000,
        help="Max tokens for evaluator LLM (default: 2000)"
    )
    
    # Workflow configuration
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations per step (default: 10)"
    )
    parser.add_argument(
        "--include-goal",
        type=str,
        choices=["first", "always", "never"],
        default="first",
        help="When to include goals in prompts: 'first' (first step only), 'always' (all steps), 'never' (no steps) (default: first)"
    )
    parser.add_argument(
        "--include-tactic",
        type=str,
        choices=["first", "always", "never"],
        default="first",
        help="When to include tactics in prompts: 'first' (first step only), 'always' (all steps), 'never' (no steps) (default: first)"
    )
    parser.add_argument(
        "--include-prerequisites",
        type=str,
        choices=["first", "always", "never"],
        default="always",
        help="When to include prerequisites in prompts: 'first' (first step only), 'always' (all steps), 'never' (no steps) (default: always)"
    )
    parser.add_argument(
        "--history-context",
        type=str,
        default="goal,command,output,results",
        help="Comma-separated list of fields to include in context history: goal, command, output, results (default: all)"
    )
    parser.add_argument(
        "--evaluation-protocol",
        type=str,
        choices=["match_alternatives", "single_path"],
        default="match_alternatives",
        help="Evaluation protocol: 'match_alternatives' (match against all alternatives), 'single_path' (validate against gold standard only) (default: match_alternatives)"
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=["command", "anticipated_result", "goal"],
        default="command",
        help="Task mode: 'command' (predict commands), 'anticipated_result' (predict information/state changes), 'goal' (predict step goal) (default: command)"
    )
    parser.add_argument(
        "--writeups-path",
        type=str,
        default="./data",
        help="Path to writeups directory (default: ./data/agentbeats)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (default: data/agentbeats/{challenge}/{challenge}_evaluation_results.json)"
    )
    
    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: from environment OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for API (default: from environment OPENAI_BASE_URL)"
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-phoenix",
        action="store_true",
        help="Disable Phoenix tracing"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_args()
    
    # Validate incompatible option combinations
    if args.task_mode == "goal" and args.include_goal == "always":
        print("Error: When task_mode is 'goal', --include-goal cannot be 'always'")
        print("       (Otherwise the agent would always be given the answer it should predict)")
        print("       Use 'never' for full challenge, or 'first' to show one example then test")
        sys.exit(1)
    
    # Determine challenge list
    if args.challenges:
        if args.challenges.lower() == "all":
            challenge_names = discover_all_challenges(args.writeups_path)
            if not challenge_names:
                print(f"Error: No challenges found in {args.writeups_path}")
                sys.exit(1)
        else:
            challenge_names = [c.strip() for c in args.challenges.split(",")]
    elif args.challenge:
        challenge_names = [args.challenge]
    else:
        print("Error: Either --challenge or --challenges must be specified")
        sys.exit(1)
    
    # Validate challenges exist
    for challenge_name in challenge_names:
        challenge_path = Path(args.writeups_path) / challenge_name
        if not challenge_path.exists():
            print(f"Error: Challenge directory not found: {challenge_path}")
            sys.exit(1)
        
        steps_file = challenge_path / "steps_enriched.json"
        if not steps_file.exists():
            print(f"Error: Steps file not found: {steps_file}")
            sys.exit(1)
    
    # Set up API credentials
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        print("Error: API key not provided. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Determine agent mode
    if args.agent_url:
        agent_mode = "a2a"
        agent_config = {
            "mode": "a2a",
            "agent_url": args.agent_url,
            "timeout": args.agent_timeout,
            "evaluation_protocol": args.evaluation_protocol,
            "task_mode": args.task_mode
        }
    else:
        agent_mode = "internal"
        agent_config = {
            "mode": "internal",
            "model": args.agent_model,
            "temperature": args.agent_temperature,
            "max_tokens": args.agent_max_tokens,
            "api_key": api_key,
            "base_url": base_url,
            "evaluation_protocol": args.evaluation_protocol,
            "task_mode": args.task_mode
        }
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if len(challenge_names) == 1:
            challenge_path = Path(args.writeups_path) / challenge_names[0]
            output_path = str(challenge_path / f"{challenge_names[0].lower()}_evaluation_results.json")
        else:
            output_path = str(Path(args.writeups_path) / "batch_evaluation_results.json")
    
    print("=" * 70)
    print("LangGraph Evaluator Workflow")
    print("=" * 70)
    if len(challenge_names) == 1:
        print(f"Challenge: {challenge_names[0]}")
    else:
        print(f"Challenges: {len(challenge_names)} ({', '.join(challenge_names)})")
    print(f"Agent Mode: {agent_mode}")
    if agent_mode == "internal":
        print(f"Agent Model: {args.agent_model}")
    else:
        print(f"Agent URL: {args.agent_url}")
    print(f"Evaluator Model: {args.evaluator_model}")
    print(f"Max Iterations per Step: {args.max_iterations}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()
    
    try:
        # Create agent interface
        print(f"Initializing agent interface ({agent_mode} mode)...")
        agent_interface = create_agent_interface(agent_config)
        
        # Create step evaluator
        print("Initializing step evaluator...")
        step_evaluator = StepEvaluator(
            model=args.evaluator_model,
            max_tokens=args.evaluator_max_tokens,
            api_key=api_key,
            base_url=base_url,
            evaluation_protocol=args.evaluation_protocol,
            task_mode=args.task_mode
        )
        
        # Create workflow
        print("Building evaluation workflow...")
        
        # Parse history context fields
        history_context_fields = [field.strip() for field in args.history_context.split(",")]
        
        workflow = EvaluatorWorkflow(
            agent_interface=agent_interface,
            step_evaluator=step_evaluator,
            max_iterations_per_step=args.max_iterations,
            enable_phoenix=not args.no_phoenix,
            include_goal=args.include_goal,
            include_tactic=args.include_tactic,
            include_prerequisites=args.include_prerequisites,
            history_context=history_context_fields
        )
        
        # Run evaluation (single or batch)
        print("Starting evaluation...\n")
        
        if len(challenge_names) == 1:
            # Single challenge evaluation
            results = workflow.run(
                challenge_name=challenge_names[0],
                agent_llm_config={
                    "model": args.agent_model,
                    "temperature": args.agent_temperature,
                    "max_tokens": args.agent_max_tokens
                },
                evaluator_llm_config={
                    "model": args.evaluator_model,
                    "max_tokens": args.evaluator_max_tokens
                }
            )
        else:
            # Batch evaluation
            results = workflow.run_batch(
                challenge_names=challenge_names,
                agent_llm_config={
                    "model": args.agent_model,
                    "temperature": args.agent_temperature,
                    "max_tokens": args.agent_max_tokens
                },
                evaluator_llm_config={
                    "model": args.evaluator_model,
                    "max_tokens": args.evaluator_max_tokens
                }
            )
        
        # Save results
        print(f"\nSaving results to {output_path}...")
        save_evaluation_results(results, output_path)
        
        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        
        if len(challenge_names) == 1:
            print(f"Final Score: {results['score']:.2%}")
            print(f"Challenge: {results['challenge']}")
        else:
            print(f"Overall Score: {results['overall_score']:.2%}")
            print(f"Challenges Evaluated: {results['challenges_evaluated']}")
            print(f"Challenges Successful: {results.get('challenges_successful', 0)}")
        
        print(f"Timestamp: {results['timestamp']}")
        print(f"Results saved to: {output_path}")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

