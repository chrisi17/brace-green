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

echo "✅ Data repository cloned successfully!"
echo ""

# Verify data structure
echo "🔍 Verifying challenge data..."

# Check for challenges with steps.json files (actual challenge directories)
CHALLENGE_DIRS=$(find "$DATA_DIR" -maxdepth 2 -type f -name "steps.json" -exec dirname {} \; | sort)
CHALLENGE_COUNT=$(echo "$CHALLENGE_DIRS" | grep -v "^$" | wc -l)

if [ "$CHALLENGE_COUNT" -gt 0 ]; then
    echo "✅ Found $CHALLENGE_COUNT challenge(s) with steps.json:"
    echo "$CHALLENGE_DIRS" | while read -r dir; do
        if [ -n "$dir" ]; then
            basename "$dir" | sed 's/^/   - /'
        fi
    done
else
    echo "⚠️  Warning: No challenges found with steps.json files"
    echo "   Searched in: $DATA_DIR"
    echo "   Looking for directories containing steps.json files"
fi

echo ""
echo "🚀 Starting application..."
echo ""

# Execute the main command
exec "$@"

