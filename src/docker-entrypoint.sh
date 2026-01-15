#!/bin/bash
set -e

# Configuration
DATA_REPO_URL="${DATA_REPO_URL:-https://github.com/LSX-UniWue/brace-ctf-data.git}"
DATA_REF="${DATA_REF:-${DATA_BRANCH:-main}}"  # Support both DATA_REF and legacy DATA_BRANCH
DATA_DIR="/home/agent/data"
SKIP_DATA_CLONE="${SKIP_DATA_CLONE:-false}"

if [ "$SKIP_DATA_CLONE" = "true" ]; then
    echo "⏭️  Skipping data repository clone/update (SKIP_DATA_CLONE=true)"
    echo "🚀 Starting server..."
    exec uv run python -m src.server "$@"
fi

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

# If data directory exists and is a git repo, update it
if [ -d "$DATA_DIR/.git" ]; then
    echo "📦 Updating existing data repository..."
    cd "$DATA_DIR"
    git fetch origin --tags
    # Check if DATA_REF is a tag or branch
    if git show-ref --tags "refs/tags/$DATA_REF" > /dev/null 2>&1; then
        echo "Checking out tag: $DATA_REF"
        git checkout "tags/$DATA_REF"
    else
        echo "Checking out branch: $DATA_REF"
        git reset --hard "origin/$DATA_REF"
    fi
    cd /home/agent
else
    # Clone fresh
    echo "📥 Cloning data repository..."
    rm -rf "$DATA_DIR"
    if ! git clone --depth 1 --branch "$DATA_REF" "$AUTH_DATA_REPO_URL" "$DATA_DIR"; then
        echo "❌ Error: Failed to clone data repository from $AUTH_DATA_REPO_URL"
        echo "   Please ensure the repository exists and is accessible, and that DATA_REPO_URL and DATA_REF are correct."
        echo "   If it's a private repository, ensure GITHUB_TOKEN or an embedded token is provided."
        exit 1
    fi
fi

# Capture commit information for evaluation tracking
cd "$DATA_DIR"
DATA_COMMIT_SHA=$(git rev-parse HEAD)
DATA_COMMIT_SHORT=$(git rev-parse --short HEAD)
DATA_COMMIT_DESCRIBE=$(git describe --tags --always 2>/dev/null || echo "$DATA_COMMIT_SHORT")
DATA_COMMIT_DATE=$(git log -1 --format=%ci)

# Extract repo owner and name from URL for condensed identifier
REPO_PATH=$(echo "$DATA_REPO_URL" | sed -E 's|.*github\.com[/:]([^/]+/[^/.]+)(\.git)?.*|\1|')
DATA_VERSION_FULL="${REPO_PATH}@${DATA_COMMIT_DESCRIBE}"

cd /home/agent

# Export as environment variables for the application to use
export DATA_REPO_URL
export DATA_COMMIT_SHA
export DATA_COMMIT_SHORT
export DATA_COMMIT_DESCRIBE
export DATA_COMMIT_DATE
export DATA_VERSION="$DATA_COMMIT_DESCRIBE"
export DATA_VERSION_FULL

echo "✅ Data repository ready!"
echo "📍 Data Version: $DATA_VERSION_FULL"
echo "   Commit SHA: $DATA_COMMIT_SHA"
echo "   Commit Date: $DATA_COMMIT_DATE"
echo ""

# Execute the server with provided arguments
echo "🚀 Starting server..."
exec uv run python -m src.server "$@"
