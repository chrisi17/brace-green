#!/bin/bash
# Docker entrypoint for white agent
# Reads environment variables and builds CLI arguments

set -e

# Build command with arguments from environment variables or defaults
CMD="uv run white_agent/server.py"

# Add host (default: 0.0.0.0 for Docker)
CMD="$CMD --host ${WHITE_AGENT_HOST:-0.0.0.0}"

# Add port (default: 8000)
CMD="$CMD --port ${WHITE_AGENT_PORT:-8000}"

# Add model (default: gpt-4o)
CMD="$CMD --model ${WHITE_AGENT_MODEL:-gpt-4o}"

# Add temperature (default: 0.7)
CMD="$CMD --temperature ${WHITE_AGENT_TEMPERATURE:-0.7}"

# Add max-tokens (default: 500)
CMD="$CMD --max-tokens ${WHITE_AGENT_MAX_TOKENS:-500}"

# Add task-mode (default: command)
CMD="$CMD --task-mode ${WHITE_AGENT_TASK_MODE:-command}"

# Add card-url if provided
if [ -n "$WHITE_AGENT_CARD_URL" ]; then
    CMD="$CMD --card-url $WHITE_AGENT_CARD_URL"
fi

echo "Starting white agent with: $CMD"
exec $CMD

