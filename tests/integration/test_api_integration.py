"""
Integration tests for the API endpoints.
"""

import asyncio
import subprocess
import sys
import time
from collections.abc import AsyncGenerator

import httpx
import pytest


@pytest.fixture(scope="session")
def api_server():
    """Start the API server for integration tests."""
    # Start the server in a subprocess
    process = subprocess.Popen(
        [
            sys.executable,  # Use the same python that runs pytest
            "-m",
            "uvicorn",
            "src.api.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",  # Use different port to avoid conflicts
            "--log-level",
            "error",  # Reduce log noise
        ]
    )
    # Give the server time to start
    time.sleep(5)
    yield "http://localhost:8001"
    # Teardown: stop the server
    process.terminate()
    process.wait()


@pytest.fixture
async def async_client(api_server) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for testing."""
    async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
        yield client


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIIntegration:
    """Integration tests for the API."""

    @pytest.mark.integration
    async def test_health_endpoint(self, async_client: httpx.AsyncClient):
        """Test health endpoint integration."""
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "version" in data
        assert "models_loaded" in data

    @pytest.mark.integration
    async def test_root_endpoint(self, async_client: httpx.AsyncClient):
        """Test root endpoint integration."""
        response = await async_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "MLOps Template API"
        assert data["version"] == "1.0.0"

    @pytest.mark.integration
    async def test_models_endpoints(self, async_client: httpx.AsyncClient):
        """Test model management endpoints."""
        # List models (should be empty initially)
        response = await async_client.get("/api/v1/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)

        # Create default model
        response = await async_client.post("/api/v1/models/default/create")
        assert response.status_code == 200

        # List models again (should now have default model)
        response = await async_client.get("/api/v1/models")
        assert response.status_code == 200
        models = response.json()
        assert "default" in models

        # Get model info
        response = await async_client.get("/api/v1/models/default")
        assert response.status_code == 200
        model_info = response.json()
        assert model_info["name"] == "default"
        assert "features" in model_info
        assert "type" in model_info

    @pytest.mark.integration
    async def test_prediction_workflow(self, async_client: httpx.AsyncClient):
        """Test complete prediction workflow."""
        # Create default model first
        response = await async_client.post("/api/v1/models/default/create")
        assert response.status_code == 200

        # Make a prediction
        prediction_request = {
            "features": [[5.1, 3.5, 1.4, 0.2]],  # Single prediction in batch format
            "model_name": "default",
            "return_probabilities": True,
        }

        response = await async_client.post("/api/v1/predict", json=prediction_request)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "probabilities" in data
        assert "model_name" in data
        assert "status" in data
        assert data["model_name"] == "default"
        assert data["status"] == "success"

        # Test batch prediction
        batch_request = {
            "features": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]],
            "model_name": "default",
            "return_probabilities": False,
        }

        response = await async_client.post("/api/v1/predict/batch", json=batch_request)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 2  # Two predictions

    @pytest.mark.integration
    async def test_error_handling(self, async_client: httpx.AsyncClient):
        """Test API error handling."""
        # Test prediction with non-existent model
        prediction_request = {
            "features": [[1.0, 2.0, 3.0, 4.0]],  # Correct batch format
            "model_name": "nonexistent",
        }

        response = await async_client.post("/api/v1/predict", json=prediction_request)
        assert response.status_code == 404

        # Test invalid prediction request
        invalid_request = {
            "features": [],  # Empty features
            "model_name": "default",
        }

        response = await async_client.post("/api/v1/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error

        # Test getting info for non-existent model
        response = await async_client.get("/api/v1/models/nonexistent")
        assert response.status_code == 404

    @pytest.mark.integration
    async def test_openapi_docs(self, async_client: httpx.AsyncClient):
        """Test OpenAPI documentation endpoints."""
        # Test OpenAPI schema
        response = await async_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "MLOps Template API"

        # Test docs endpoint
        response = await async_client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.integration
    async def test_concurrent_predictions(self, async_client: httpx.AsyncClient):
        """Test concurrent predictions."""
        # Create default model first
        response = await async_client.post("/api/v1/models/default/create")
        assert response.status_code == 200

        # Make multiple concurrent predictions
        prediction_request = {
            "features": [[5.1, 3.5, 1.4, 0.2]],
            "model_name": "default",
        }

        # Create multiple concurrent requests
        tasks = []
        for _ in range(10):
            task = async_client.post("/api/v1/predict", json=prediction_request)
            tasks.append(task)

        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks)

        # Check all responses are successful
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data


@pytest.mark.integration
@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for the API."""

    @pytest.mark.integration
    async def test_prediction_performance(self, async_client: httpx.AsyncClient):
        """Test prediction endpoint performance."""
        # Create default model
        response = await async_client.post("/api/v1/models/default/create")
        assert response.status_code == 200

        prediction_request = {
            "features": [[5.1, 3.5, 1.4, 0.2]],
            "model_name": "default",
        }

        # Measure response times
        response_times = []
        for _ in range(20):
            start_time = time.time()
            response = await async_client.post(
                "/api/v1/predict", json=prediction_request
            )
            end_time = time.time()

            assert response.status_code == 200
            response_times.append(end_time - start_time)

        # Check performance metrics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # Performance assertions (adjust thresholds as needed)
        assert avg_response_time < 1.0  # Average response time under 1 second
        assert max_response_time < 2.0  # Max response time under 2 seconds

    @pytest.mark.integration
    async def test_batch_prediction_performance(self, async_client: httpx.AsyncClient):
        """Test batch prediction performance."""
        # Create default model
        response = await async_client.post("/api/v1/models/default/create")
        assert response.status_code == 200

        # Test with different batch sizes
        batch_sizes = [1, 10, 50, 100]

        for batch_size in batch_sizes:
            batch_request = {
                "features": [[5.1, 3.5, 1.4, 0.2]] * batch_size,
                "model_name": "default",
            }

            start_time = time.time()
            response = await async_client.post(
                "/api/v1/predict/batch", json=batch_request
            )
            end_time = time.time()

            assert response.status_code == 200

            data = response.json()
            assert len(data["predictions"]) == batch_size

            # Check that batch processing is efficient
            processing_time = end_time - start_time
            assert processing_time < 5.0  # Should complete within 5 seconds
