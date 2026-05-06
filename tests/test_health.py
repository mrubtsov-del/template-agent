"""Tests for the health route."""

from fastapi.testclient import TestClient

from template_agent.src.routes.health import router


class TestHealthRoute:
    """Test cases for health endpoint."""

    def test_health_endpoint(self):
        """Test health endpoint returns correct response."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Snowflake Agent"

    def test_health_endpoint_content_type(self):
        """Test health endpoint returns correct content type."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"

    def test_shadowbot_health_alias_endpoint(self):
        """Test shadowbot-specific health alias returns healthy response."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        response = client.get("/v1/shadowbot-agent/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Snowflake Agent"
