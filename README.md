# BraceGreen - CTF Evaluator Green Agent

A green agent for evaluating CTF-solving agents on the [AgentBeats platform](https://agentbeats.dev).

## Quick Start

### Build and Run with Docker

```bash
# Build the image
docker build -t bracegreen-evaluator .

# Run the container
docker run -p 9001:9001 \
  -e OPENAI_API_KEY=your-api-key \
  -e OPENAI_BASE_URL=https://api.openai.com/v1 \
  -e DATA_REPO_URL=https://github.com/LSX-UniWue/brace-ctf-data.git \
  -e DATA_BRANCH=master \
  bracegreen-evaluator
```

### Test the Agent

```bash
# Check agent card
curl http://localhost:9001/

# Expected response: Agent card in TOML format
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI API base URL |
| `DATA_REPO_URL` | No | `https://github.com/LSX-UniWue/brace-ctf-data.git` | Git repository with challenge data |
| `DATA_BRANCH` | No | `master` | Branch to use from data repository |
| `PORT` | No | `9001` | Port for the agent server |

## How It Works

1. **Container starts** → Entrypoint script clones the latest challenge data from the configured Git repository
2. **Data is cached** → Subsequent restarts pull only the changes
3. **Agent serves** → A2A-compatible green agent server starts on port 9001
4. **Always fresh** → No need to rebuild the image when challenge data updates

## AgentBeats Deployment

This agent is designed to be deployed on [AgentBeats](https://agentbeats.dev) as a green agent evaluator.

See the [AgentBeats Tutorial](https://docs.agentbeats.dev/tutorial/) for deployment instructions.

## Docker Image

Pre-built images are available at:
```
ghcr.io/lsx-uniwue/bracegreen:latest
ghcr.io/lsx-uniwue/bracegreen:v1.0.0
```

## Architecture

- **Green Agent Server**: A2A-compatible server that orchestrates CTF evaluations
- **LangGraph Workflow**: Step-by-step evaluation with semantic comparison
- **Dynamic Data Loading**: Challenge data fetched at runtime from Git repository
- **White Agent Interface**: Evaluates CTF-solving agents via A2A protocol

## License

See [LICENSE](LICENSE) for details.

## Development

For development setup and full documentation, see the main development repository.
