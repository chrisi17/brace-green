# BraceGreen Test Suite

This directory contains tests for the BraceGreen evaluation framework.

## Test Categories

### A2A Conformance Tests (`test_agent.py`)

Official A2A protocol conformance tests based on the [RDI-Foundation/green-agent-template](https://github.com/RDI-Foundation/green-agent-template/blob/main/tests/test_agent.py).

**Purpose**: Validate that our agents conform to the A2A protocol specification.

**Included in public release**: ✅ Yes

**Run with**:
```bash
# Start the agents first
docker run -d -p 9001:9001 --name green-agent bracegreen-evaluator
docker run -d -p 8000:8000 --name white-agent bracegreen-white

# Run tests
uv run pytest tests/test_agent.py -v
```

### Internal Development Tests

These tests are used during development but **NOT** included in the public release:

#### `test_a2a_conformance.py`
- Custom A2A validation tests
- Lower-level protocol testing
- Development/debugging helpers

#### `test_prerequisite_refinement.py`
- Tests for prerequisite analysis workflow
- Internal LangGraph workflow testing

#### `test_evaluator_integration.py`
- End-to-end integration tests
- Requires local setup and data

#### `conftest.py`
- Shared test fixtures
- Helper functions for development tests

## Running Tests

### Development (all tests)

```bash
# Install test dependencies
uv sync --extra test

# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_agent.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### CI/CD (conformance only)

The public repository only includes `test_agent.py` for A2A conformance validation in GitHub Actions.

## Test Requirements

### A2A Conformance Tests
- Running green agent on port 9001
- Running white agent on port 8000 (for white agent tests)
- `a2a` Python package installed
- `pytest` and `pytest-asyncio`

### Internal Tests
- Full development environment
- Local challenge data in `data/agentbeats/`
- OpenAI API key configured
- All project dependencies

## Writing New Tests

### For Public A2A Conformance
Add tests to `test_agent.py` following the pattern from the [official template](https://github.com/RDI-Foundation/green-agent-template/blob/main/tests/test_agent.py).

### For Internal Development
Create new test files with descriptive names (e.g., `test_feature_name.py`).

## See Also

- [A2A Protocol Specification](https://a2aproject.org/)
- [Green Agent Template](https://github.com/RDI-Foundation/green-agent-template)
- [AgentBeats Documentation](https://docs.agentbeats.dev/)

