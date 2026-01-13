# Dockerfile for BraceGreen Green Agent
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install git for data cloning
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Copy entrypoint script for data loading (with execute permissions)
COPY --chmod=755 docker-entrypoint.sh /home/agent/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_REPO_URL=https://github.com/LSX-UniWue/brace-ctf-data.git
ENV DATA_BRANCH=master

ENTRYPOINT ["/home/agent/docker-entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "9001", "--writeups-path", "/home/agent/data"]
EXPOSE 9001

