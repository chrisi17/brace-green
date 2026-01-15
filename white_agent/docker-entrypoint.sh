#!/bin/bash
# Docker entrypoint for white agent
# Reads environment variables and/or command-line arguments and builds CLI arguments

set -e

# Build command with arguments from environment variables or defaults
CMD="uv run white_agent/server.py"

# Parse command-line arguments if provided, otherwise use environment variables
# Command-line arguments take precedence over environment variables

# Parse arguments
HOST="${WHITE_AGENT_HOST:-0.0.0.0}"
PORT="${WHITE_AGENT_PORT:-8000}"
MODEL="${WHITE_AGENT_MODEL:-gpt-5.1}"
TEMPERATURE="${WHITE_AGENT_TEMPERATURE:-0.7}"
MAX_TOKENS="${WHITE_AGENT_MAX_TOKENS:-500}"
TASK_MODE="${WHITE_AGENT_TASK_MODE:-command}"
CARD_URL="${WHITE_AGENT_CARD_URL:-}"
MOCK_MODE="${WHITE_AGENT_MOCK_MODE:-false}"
VERBOSE="${WHITE_AGENT_VERBOSE:-false}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --task-mode)
            TASK_MODE="$2"
            shift 2
            ;;
        --card-url)
            CARD_URL="$2"
            shift 2
            ;;
        --mock-mode)
            MOCK_MODE="true"
            shift
            ;;
        *)
            # Unknown argument, pass through
            CMD="$CMD $1"
            shift
            ;;
    esac
done

# Build final command
CMD="$CMD --host $HOST"
CMD="$CMD --port $PORT"
CMD="$CMD --model $MODEL"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --max-tokens $MAX_TOKENS"
CMD="$CMD --task-mode $TASK_MODE"

# Add card-url if provided
if [ -n "$CARD_URL" ]; then
    CMD="$CMD --card-url $CARD_URL"
fi

# Add mock-mode flag if enabled
if [ "$MOCK_MODE" = "true" ]; then
    CMD="$CMD --mock-mode"
fi

echo "Starting white agent with: $CMD"
echo "Environment: VERBOSE=$VERBOSE"
exec $CMD

