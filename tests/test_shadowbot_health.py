"""Tests for the dedicated Shadowbot health endpoint (/v1/shadowbot-agent/health).

This route lives in template_agent/src/routes/shadowbot_health.py and is what
the Red Hat AI Studio / Shadowbot validator hits during agent registration.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from template_agent.src.routes.shadowbot_health import router


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestShadowbotHealth:
    def test_returns_200_and_required_payload(self):
        response = _make_client().get("/v1/shadowbot-agent/health")
        assert response.status_code == 200

        data = response.json()
        # Schema required by Shadowbot validator.
        assert data["status"] == "healthy"
        assert data["service"] == "Snowflake Agent"
        assert data["version"] == "1.0.0"

    def test_content_type_is_json(self):
        response = _make_client().get("/v1/shadowbot-agent/health")
        assert response.headers["content-type"] == "application/json"
