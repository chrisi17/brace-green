#!/bin/bash
set -e

# Configuration
DATA_REPO_URL="${DATA_REPO_URL:-https://github.com/yourusername/bracegreen-data.git}"
DATA_BRANCH="${DATA_BRANCH:-main}"
DATA_DIR="/app/data"

echo "🔄 Fetching latest challenge data..."

# If data directory exists and is a git repo, pull latest
if [ -d "$DATA_DIR/.git" ]; then
    echo "📦 Updating existing data repository..."
    cd "$DATA_DIR"
    git fetch origin
    git reset --hard "origin/$DATA_BRANCH"
    cd /app
else
    # Clone fresh
    echo "📥 Cloning data repository..."
    rm -rf "$DATA_DIR"
    git clone --depth 1 --branch "$DATA_BRANCH" "$DATA_REPO_URL" "$DATA_DIR"
fi

echo "✅ Data updated successfully!"
echo ""

# Execute the main command
exec "$@"

