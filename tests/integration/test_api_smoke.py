from fastapi.testclient import TestClient

from caption_agent.api.main import app


def test_health_smoke() -> None:
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200

