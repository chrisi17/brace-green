import httpx
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9001",
        help="Agent URL (default: http://localhost:9001 for green agent)",
    )


@pytest.fixture(scope="session")
def agent(request):
    """Agent URL fixture. Agent must be running before tests start."""
    url = request.config.getoption("--agent-url")

    # Try to connect with retries (agent might still be starting)
    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=5)
            if response.status_code == 200:
                return url
            else:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
                pytest.exit(f"Agent at {url} returned status {response.status_code}", returncode=1)
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
                continue
            pytest.exit(f"Could not connect to agent at {url} after {max_retries} attempts: {e}", returncode=1)

    return url
