# Prerequisite Refinement Module

This module provides tools for analyzing and refining CTF challenge prerequisites by identifying information gaps between what agents have access to and what they need to know.

## Purpose

The **Prerequisite Refinement Workflow** helps improve challenge quality by:
1. Analyzing what context is available to agents (via `build_step_context`)
2. Comparing against the gold standard requirements
3. Identifying missing information that agents cannot reasonably infer
4. Suggesting refined prerequisites with source attribution

## Module Structure

```
src/prerequisite_refinement/
├── __init__.py              # Module exports
├── analyzer.py              # PrerequisiteAnalyzer (LLM-based gap analysis)
├── workflow.py              # PrerequisiteRefinementWorkflow (LangGraph)
└── README.md                # This file
```

## Components

### PrerequisiteAnalyzer (`analyzer.py`)

LLM-based semantic analyzer that identifies missing information in prerequisites.

**Key Methods:**
- `analyze_step()` - Main analysis method
- `_build_analysis_prompt()` - Constructs comprehensive analysis prompt  
- `_parse_analysis_response()` - Parses JSON response from LLM
- `from_config()` - Factory method for configuration-based instantiation

**Features:**
- Identifies concrete missing facts (IPs, usernames, passwords, paths, etc.)
- Assigns criticality levels (HIGH/MEDIUM/LOW) to each gap
- Tracks where missing information should come from (previous steps, environment, etc.)
- Suggests refined prerequisites

### PrerequisiteRefinementWorkflow (`workflow.py`)

LangGraph state machine that orchestrates the prerequisite analysis process.

**Workflow Nodes:**
1. `load_challenge` - Load challenge steps and initialize state
2. `prepare_step` - Prepare for analyzing the next step
3. `analyze_prerequisites` - Perform gap analysis on current step
4. `record_analysis` - Record results and move to next step
5. `finalize` - Format final output

**Features:**
- Step-by-step analysis with state management
- Uses same context building as main evaluation workflow
- Phoenix tracing integration
- Checkpoint support via MemorySaver
- Teacher forcing (gold standard context history)
- Dynamic recursion limit calculation

## Usage

### Basic Usage

```python
from prerequisite_refinement import PrerequisiteAnalyzer, PrerequisiteRefinementWorkflow

# Create analyzer
analyzer = PrerequisiteAnalyzer(
    model="gpt-4o",
    max_tokens=3000
)

# Create workflow
workflow = PrerequisiteRefinementWorkflow(
    prerequisite_analyzer=analyzer,
    enable_phoenix=True
)

# Run analysis
results = workflow.run(
    challenge_name="Funbox",
    analyzer_llm_config={
        "model": "gpt-4o",
        "max_tokens": 3000
    }
)

# results contains:
# {
#   "challenge": "Funbox",
#   "total_steps": 13,
#   "analyses": [ ... ]  # One analysis per step
# }
```

### Command Line

```bash
# Using uv (recommended)
uv run examples/run_prerequisite_refinement.py --challenge Funbox

# With custom model
uv run examples/run_prerequisite_refinement.py \
  --challenge Funbox \
  --model gpt-4o-mini

# Disable Phoenix tracing
uv run examples/run_prerequisite_refinement.py \
  --challenge Funbox \
  --disable-phoenix
```

## Output Format

### JSON Output

```json
{
  "challenge": "Funbox",
  "total_steps": 13,
  "analyses": [
    {
      "step_index": 0,
      "goal": "Identify the target IP address on the network",
      "tactic": "Reconnaissance",
      "analysis": {
        "current_prerequisites": [
          "Attacker machine on same network as target",
          "netdiscover tool available"
        ],
        "missing_information": [
          {
            "item": "Network subnet to scan (192.168.0.0/24)",
            "criticality": "high",
            "reason": "Cannot determine which subnet to scan without prior knowledge",
            "source": "Should be provided as prerequisite or environment setup"
          }
        ],
        "suggested_prerequisites": [
          "Attacker machine on same network as target",
          "netdiscover tool available",
          "Target is on the 192.168.0.0/24 subnet"
        ],
        "analysis_explanation": "The agent needs to know which network subnet to scan..."
      }
    }
  ]
}
```

### Text Report

Human-readable report with:
- Summary statistics
- Per-step analysis with sections for:
  - Current prerequisites
  - Missing information (with criticality markers)
  - Suggested refined prerequisites
  - Analysis explanation
- Criticality breakdown in summary

## Integration

### Shared Components

The prerequisite refinement workflow reuses components from the main evaluation workflow:

**From `src/evaluator/`:**
- `state.py` - EvaluationState TypedDict
- `utils.py` - Context building (`build_step_context`), step loading (`load_challenge_steps`)

This ensures consistency and reduces code duplication while maintaining separation of concerns.

### Independence

The prerequisite refinement workflow is **completely independent** from the main evaluation workflow:
- No shared workflow nodes
- Separate entry points
- Different execution patterns
- Can be run standalone without the evaluation workflow

## Configuration

### Analyzer Configuration

```python
analyzer_config = {
    "model": "gpt-4o",           # LLM model to use
    "max_tokens": 3000,          # Maximum response tokens
    "api_key": None,             # Falls back to OPENAI_API_KEY env var
    "base_url": None             # Falls back to OPENAI_BASE_URL env var
}

analyzer = PrerequisiteAnalyzer.from_config(analyzer_config)
```

### Environment Variables

```bash
# OpenAI API Configuration
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Optional

# Phoenix Tracing
export PHOENIX_PROJECT_NAME="bracegreen-prerequisite-refinement"
export PHOENIX_ENDPOINT="localhost:4317"  # Optional
```

## Development

### Running Tests

```bash
# Run test suite
uv run tests/test_prerequisite_refinement.py

# Note: Tests make real API calls (not mocked)
# Make sure OPENAI_API_KEY is set
```

### Adding New Features

1. **New Analysis Capabilities**: Extend `PrerequisiteAnalyzer` methods
2. **Workflow Modifications**: Update nodes in `PrerequisiteRefinementWorkflow`
3. **Output Formats**: Modify `format_analysis_report()` in CLI script

### Debugging

Enable Phoenix tracing for detailed observability:

```bash
# Terminal 1: Start Phoenix
phoenix serve

# Terminal 2: Run analysis with tracing
uv run examples/run_prerequisite_refinement.py --challenge Funbox

# View traces at http://localhost:6006
```

## Comparison with Evaluation Workflow

| Feature | Evaluation Workflow | Prerequisite Refinement |
|---------|-------------------|------------------------|
| **Location** | `src/evaluator/` | `src/prerequisite_refinement/` |
| **Purpose** | Test agent performance | Improve challenge quality |
| **Input** | Challenge + Agent | Challenge only |
| **Output** | Pass/fail scores | Gap analysis |
| **Complexity** | Iterative (subgraph) | Linear (single-pass) |
| **Time/Step** | ~30s - 5min | ~5-15s |
| **Use Case** | CI/CD testing | Challenge authoring |

## See Also

- [Main Documentation](../../PREREQUISITE_REFINEMENT.md) - Complete documentation
- [Architecture Diagram](../../PREREQUISITE_REFINEMENT_DIAGRAM.md) - Visual diagrams
- [Examples](../../examples/run_prerequisite_refinement.py) - CLI script
- [Tests](../../tests/test_prerequisite_refinement.py) - Test suite


