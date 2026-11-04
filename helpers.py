"""Small helpers shared by the test files."""

from fastapi.testclient import TestClient


def register(client: TestClient, username: str = "alice", password: str = "correct-horse-9"):
    """Register a user (this also logs them in)."""
    return client.post("/auth/register", json={"username": username, "password": password})
