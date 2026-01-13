#!/bin/bash
set -e

# Configuration
DATA_REPO_URL="${DATA_REPO_URL:-https://github.com/LSX-UniWue/brace-ctf-data.git}"
DATA_BRANCH="${DATA_BRANCH:-main}"
DATA_DIR="/home/agent/data"

echo "🔄 Fetching latest challenge data..."

# Prepare authenticated URL if GITHUB_TOKEN is available and not already embedded
AUTH_DATA_REPO_URL="$DATA_REPO_URL"
if [[ "$DATA_REPO_URL" != *"@"* ]] && [[ -n "$GITHUB_TOKEN" ]]; then
    echo "Injecting GITHUB_TOKEN for authentication."
    AUTH_DATA_REPO_URL=$(echo "$DATA_REPO_URL" | sed "s|https://|https://x-access-token:$GITHUB_TOKEN@|")
elif [[ "$DATA_REPO_URL" == *"@"* ]]; then
    echo "Using DATA_REPO_URL with embedded token."
else
    echo "No authentication token found. Assuming public repository."
fi

# If data directory exists and is a git repo, pull latest
if [ -d "$DATA_DIR/.git" ]; then
    echo "📦 Updating existing data repository..."
    cd "$DATA_DIR"
    git fetch origin
    git reset --hard "origin/$DATA_BRANCH"
    cd /home/agent
else
    # Clone fresh
    echo "📥 Cloning data repository..."
    rm -rf "$DATA_DIR"
    if ! git clone --depth 1 --branch "$DATA_BRANCH" "$AUTH_DATA_REPO_URL" "$DATA_DIR"; then
        echo "❌ Error: Failed to clone data repository from $AUTH_DATA_REPO_URL"
        echo "   Please ensure the repository exists and is accessible, and that DATA_REPO_URL and DATA_BRANCH are correct."
        echo "   If it's a private repository, ensure GITHUB_TOKEN or an embedded token is provided."
        exit 1
    fi
fi

echo "✅ Data repository ready!"
echo ""

# Execute the server with provided arguments
echo "🚀 Starting server..."
exec uv run python -m src.server "$@"
