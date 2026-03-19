import pytest
import httpx
import json
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


MOCK_ROUTES = {
    "model-a": "http://model-a:8000",
    "model-b": "http://model-b:8000",
}


@pytest.fixture
def client():
    with patch("gateway.main.load_routes", return_value=MOCK_ROUTES.copy()):
        from gateway.main import create_app
        app = create_app()
        yield TestClient(app)


class TestModelsEndpoint:
    def test_models_returns_list(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        model_ids = [m["id"] for m in data["data"]]
        assert "model-a" in model_ids
        assert "model-b" in model_ids


class TestHealthEndpoint:
    def test_health_all_healthy(self, client):
        mock_response = httpx.Response(200, json={"status": "ok"})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    def test_health_partial_down_returns_degraded(self, client):
        async def mock_get(url):
            if "model-a" in url:
                return httpx.Response(200, json={"status": "ok"})
            raise httpx.ConnectError("Connection refused")

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=mock_get):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "degraded"
            assert response.json()["models"]["model-a"] == "healthy"
            assert response.json()["models"]["model-b"] == "down"

    def test_health_all_down_returns_503(self, client):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "unhealthy"


class TestRouting:
    def test_unknown_model_returns_422(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 422
        available = response.json()["available_models"]
        assert "model-a" in available
        assert "model-b" in available

    def test_missing_model_field_returns_422(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 422

    def test_backend_down_returns_503(self, client):
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 503
            assert "unavailable" in response.json()["error"]

    def test_proxy_forwards_to_backend(self, client):
        mock_resp = httpx.Response(200, json={"id": "chatcmpl-123", "choices": [{"message": {"content": "hello"}}]})
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            response = client.post(
                "/v1/chat/completions",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 200
            assert response.json()["id"] == "chatcmpl-123"
