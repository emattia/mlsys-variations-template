import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    """Ensure that the health check endpoint returns OK and expected structure."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200, "Health endpoint should return HTTP 200"

    payload = response.json()
    # Basic contract assertions
    assert "status" in payload, "Response must include service status"
    assert payload["status"] in {"healthy", "no_models_loaded"}
    assert "timestamp" in payload, "Response must include timestamp"
    assert "version" in payload, "Response must include version"
    assert "models_loaded" in payload, "Response must include loaded models list"
    assert isinstance(payload["models_loaded"], list)
