# BraceGreen Evaluator - Source Code

A2A-compatible green agent for evaluating penetration testing agents on CTF challenges with AgentBeats.

Based on the [A2A Agent Template](https://github.com/RDI-Foundation/green-agent-template) for building AgentBeats-compatible green agents.

## Project Structure

```
src/
├─ server.py           # A2A server for AgentBeats (production mode)
├─ executor.py         # A2A request handling and task lifecycle
├─ agent.py            # Main evaluator agent implementation
├─ messenger.py        # A2A messaging utilities
├─ white_agent.py      # Baseline white agent (purple agent) under test
├─ evaluator/          # LangGraph-based evaluation workflow
│  ├─ main.py          # Standalone CLI runner (development/testing mode)
│  ├─ workflow.py      # LangGraph workflow definition
│  ├─ agent_interface.py  # Interface to agent under test
│  ├─ step_evaluator.py   # Semantic comparison of predictions
│  ├─ state.py         # State schema definitions
│  ├─ utils.py         # Helper functions
│  └─ a2a_client.py    # A2A protocol client
└─ tests/              # Unit and integration tests
   ├─ conftest.py      # Test configuration
   └─ test_agent.py    # Agent tests
```

## Data Repository

The evaluator requires CTF challenge data from the [brace-ctf-data](https://github.com/LSX-UniWue/brace-ctf-data) repository:

```bash
# Clone the challenge data repository
git clone https://github.com/LSX-UniWue/brace-ctf-data.git data
```

## Running Locally

### AgentBeats Server Mode (A2A)

Run the evaluator as an A2A server for AgentBeats integration:

```bash
# Install dependencies
uv sync

# Clone challenge data
git clone https://github.com/LSX-UniWue/brace-ctf-data.git data

# Run the A2A server (default port 9001)
uv run python -m src.server

# Or with custom configuration
uv run python -m src.server --host 0.0.0.0 --port 9001 --writeups-path ./data
```

### Standalone CLI Mode

For direct local evaluation without AgentBeats:

```bash
# See CLI Usage section below or src/evaluator/README.md for details
uv run python -m src.evaluator.main --challenge Funbox
```

## Running with Docker (AgentBeats Deployment)

The Docker container runs `src/server.py` (A2A server mode) with dynamic data loading - challenge data is cloned from Git at container startup rather than being baked into the image.

```bash
# Build the image
docker build -t bracegreen-evaluator .

# Run the A2A server (data is cloned automatically at startup)
docker run -p 9001:9001 \
  -e OPENAI_API_KEY=your-key \
  -e DATA_REPO_URL=https://github.com/LSX-UniWue/brace-ctf-data \
  -e DATA_BRANCH=master \
  bracegreen-evaluator
```

The container automatically:
1. Clones challenge data from `DATA_REPO_URL` to `/home/agent/data`
2. Starts `src/server.py` on port 9001 with `--writeups-path /home/agent/data`

**Environment Variables:**
- `OPENAI_API_KEY` - Required: OpenAI API key for LLM calls
- `DATA_REPO_URL` - Optional: Git repository URL for challenge data (default: https://github.com/LSX-UniWue/brace-ctf-data.git)
- `DATA_BRANCH` - Optional: Git branch to use (default: master)
- `GITHUB_TOKEN` - Optional: For private data repositories
- `SKIP_DATA_CLONE` - Optional: Set to "true" to skip data cloning (default: false)

**Server Configuration (passed to `src/server.py`):**

You can override the default server arguments:
```bash
docker run -p 9001:9001 \
  -e OPENAI_API_KEY=your-key \
  bracegreen-evaluator \
  --host 0.0.0.0 --port 9001 --writeups-path /home/agent/data
```

See [AgentBeats Deployment Guide](../AGENTBEATS_DEPLOYMENT.md) for complete deployment instructions.


## Testing the A2A Server

Run A2A conformance tests against the AgentBeats server:

```bash
# Install test dependencies
uv sync --extra test

# Clone challenge data
git clone https://github.com/LSX-UniWue/brace-ctf-data.git data

# Start the A2A evaluator server
uv run python -m src.server

# In another terminal, run A2A conformance tests
uv run pytest --agent-url http://localhost:9001
```

## Usage Modes

BraceGreen supports two usage modes:

### 1. AgentBeats Server Mode (Production)

Run as an A2A server for AgentBeats platform integration. The server receives evaluation requests via the A2A protocol.

**Start the server:**
```bash
uv run python -m src.server --host 0.0.0.0 --port 9091 --writeups-path ./data
```



**Send evaluation requests** to `http://localhost:9001` with this minimum configuration:

```json
{
  "participants": {
    "solver": "http://localhost:9002"
  },
  "config": {
    "challenges": ["Funbox"],
    "agent_config": {
      "mode": "a2a"
    }
  }
}
```

For details on the leaderboard and runner, please refer to [brace-agentbeats-leaderboard](https://github.com/LSX-UniWue/brace-agentbeats-leaderboard).


### 2. Standalone CLI Mode of the Workflow (Development/Testing)

Run evaluations directly from the command line without AgentBeats:

```bash
# Evaluate a single challenge
uv run python -m src.evaluator.main --challenge Funbox

# Evaluate all challenges
uv run python -m src.evaluator.main --challenges all

# Evaluate a remote A2A agent
uv run python -m src.evaluator.main --challenge Funbox --agent-url http://localhost:9002

# Custom configuration
uv run python -m src.evaluator.main \
  --challenge Funbox \
  --agent-model gpt-5.1 \
  --evaluator-model gpt-5.1 \
  --max-iterations 10 \
  --output results.json
```

See [Evaluator Documentation](evaluator/README.md) for all CLI options.


## Publishing

GitHub Actions automatically builds and publishes Docker images to GitHub Container Registry:

- **Push to `main`** → publishes `latest` tag
- **Create a git tag** (e.g. `v1.0.0`) → publishes version tags

```bash
git tag v1.0.0
git push origin v1.0.0
```

The image will be available at:
```
ghcr.io/LSX-UniWue/brace-green:latest
ghcr.io/LSX-UniWue/brace-green:1.0.0
ghcr.io/LSX-UniWue/brace-green:1
```

## Documentation

- [Main README](../README.md) - Project overview
- [Evaluator Documentation](evaluator/README.md) - Detailed workflow documentation
