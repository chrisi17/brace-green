# Dockerfile for BraceGreen Green Agent
FROM python:3.11-slim

# Install system dependencies (git for cloning data)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=9001
ENV DATA_REPO_URL=https://github.com/LSX-UniWue/brace-ctf-data.git
ENV DATA_BRANCH=master

# Expose port
EXPOSE 9001

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Run the green agent server
# Note: Writeups path is set to /app/data (data repo root) since challenges are at root level
CMD ["uv", "run", "python", "-m", "src.evaluator.main", "--serve", "--serve-host", "0.0.0.0", "--serve-port", "9001", "--writeups-path", "/app/data"]

