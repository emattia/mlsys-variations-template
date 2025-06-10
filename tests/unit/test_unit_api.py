"""
Unit tests for the API components.
"""

import time
import warnings
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.models import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.service import ModelService
from src.config.manager import ConfigManager

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but",
    category=UserWarning,
    module="sklearn",
)


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager."""
    config_manager = Mock(spec=ConfigManager)
    config = Mock()
    config.paths.model_root = "models"
    config_manager.get_config.return_value = config
    return config_manager


@pytest.fixture
def model_service(mock_config_manager):
    """Create a model service with mocked dependencies."""
    return ModelService(mock_config_manager)


@pytest.fixture
def mock_service():
    """Create a mock model service."""
    mock_service = Mock()
    mock_service.health_check.return_value = {
        "status": "no_models_loaded",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "models_loaded": [],
        "uptime_seconds": 100.0,
    }
    mock_service.list_models.return_value = []
    mock_service.create_default_model.return_value = False
    mock_service.load_model.return_value = False
    mock_service.unload_model.return_value = False
    return mock_service


@pytest.fixture
def client(mock_service):
    """Create a test client with mocked dependencies."""
    from src.api.app import create_app
    from src.api.routes import get_model_service

    app = create_app()

    # Override dependency
    def get_mock_service(request: Request) -> ModelService:
        return mock_service

    app.dependency_overrides[get_model_service] = get_mock_service

    return TestClient(app)


@pytest.fixture
def client_with_mock_service(mock_config_manager):
    """Create a test client with mocked service."""
    from src.api.routes import get_model_service

    app = create_app()

    # Create a mock service
    mock_service = Mock()
    mock_service.health_check.return_value = {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "models_loaded": ["default"],
        "uptime_seconds": 100.0,
    }
    mock_service.list_models.return_value = ["model1", "model2"]
    mock_service.predict.return_value = ([1], [[0.2, 0.8]], 15.5)
    mock_service.load_model.return_value = True
    mock_service.unload_model.return_value = True

    # Override the dependency
    def override(request: Request) -> ModelService:
        return mock_service

    app.dependency_overrides[get_model_service] = override

    return TestClient(app)


class TestModelService:
    """Test cases for ModelService."""

    def test_init(self, mock_config_manager):
        """Test ModelService initialization."""
        service = ModelService(mock_config_manager)
        assert service.config_manager == mock_config_manager
        assert service.models == {}
        assert isinstance(service.startup_time, float)

    def test_get_uptime(self, model_service):
        """Test uptime calculation."""
        time.sleep(0.1)  # Wait a bit
        uptime = model_service.get_uptime()
        assert uptime > 0
        assert uptime < 1  # Should be less than 1 second

    def test_list_models_empty(self, model_service):
        """Test listing models when none are loaded."""
        models = model_service.list_models()
        assert models == []

    def test_health_check_no_models(self, model_service):
        """Test health check with no models loaded."""
        health = model_service.health_check()
        assert health["status"] == "no_models_loaded"
        assert health["models_loaded"] == []
        assert "timestamp" in health
        assert "uptime_seconds" in health
        assert health["version"] == "1.0.0"

    @patch("joblib.load")
    @patch("pathlib.Path.exists")
    def test_load_model_success(self, mock_exists, mock_joblib_load, model_service):
        """Test successful model loading."""
        # Mock model
        mock_model = Mock()
        mock_model._estimator_type = "classifier"
        mock_model.classes_ = ["class1", "class2"]
        mock_model.feature_names_in_ = np.array(["feature1", "feature2"])
        mock_joblib_load.return_value = mock_model
        mock_exists.return_value = True

        # Test loading
        success = model_service.load_model("test_model", "test_path.joblib")

        assert success is True
        assert "test_model" in model_service.models
        assert model_service.models["test_model"]["model"] == mock_model

    def test_load_model_not_found(self, model_service):
        """Test loading non-existent model."""
        success = model_service.load_model("nonexistent")
        assert success is False

    def test_get_model_info_not_found(self, model_service):
        """Test getting info for non-existent model."""
        info = model_service.get_model_info("nonexistent")
        assert info is None

    def test_unload_model_success(self, model_service):
        """Test successful model unloading."""
        # Add a mock model
        model_service.models["test_model"] = {"model": Mock()}

        success = model_service.unload_model("test_model")
        assert success is True
        assert "test_model" not in model_service.models

    def test_unload_model_not_found(self, model_service):
        """Test unloading non-existent model."""
        success = model_service.unload_model("nonexistent")
        assert success is False

    @patch("sklearn.ensemble.RandomForestClassifier")
    @patch("sklearn.datasets.make_classification")
    @patch("joblib.dump")
    @patch("pathlib.Path.mkdir")
    def test_create_default_model(
        self, mock_mkdir, mock_dump, mock_make_classification, mock_rf, model_service
    ):
        """Test creating default model."""
        # Mock training data
        mock_make_classification.return_value = (
            np.random.rand(100, 4),
            np.random.randint(0, 2, 100),
        )

        # Mock model
        mock_model = Mock()
        mock_model.feature_names_in_ = np.array(
            ["feature_0", "feature_1", "feature_2", "feature_3"]
        )
        mock_rf.return_value = mock_model

        with patch.object(model_service, "load_model", return_value=True):
            success = model_service.create_default_model()
            assert success is True


class TestPydanticModels:
    """Test cases for Pydantic models."""

    def test_prediction_request_valid(self):
        """Test valid prediction request."""
        request = PredictionRequest(
            features=[1.0, 2.0, 3.0, 4.0],
            model_name="test_model",
            return_probabilities=True,
        )
        assert request.features == [1.0, 2.0, 3.0, 4.0]
        assert request.model_name == "test_model"
        assert request.return_probabilities is True

    def test_prediction_request_defaults(self):
        """Test prediction request with defaults."""
        request = PredictionRequest(features=[1.0, 2.0])
        assert request.model_name == "default"
        assert request.return_probabilities is False

    def test_prediction_request_empty_features(self):
        """Test prediction request with empty features."""
        with pytest.raises(ValueError):
            PredictionRequest(features=[])

    def test_prediction_response_valid(self):
        """Test valid prediction response."""
        response = PredictionResponse(
            prediction=[1],
            probabilities=[[0.2, 0.8]],
            model_name="test_model",
            timestamp=datetime.now(),
            processing_time_ms=15.5,
        )
        assert response.prediction == [1]
        assert response.probabilities == [[0.2, 0.8]]
        assert response.model_name == "test_model"

    def test_health_response_valid(self):
        """Test valid health response."""
        response = HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            models_loaded=["model1", "model2"],
            uptime_seconds=123.45,
        )
        assert response.status == "healthy"
        assert response.models_loaded == ["model1", "model2"]


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "MLOps Template API"
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        # Mock service is automatically injected via fixture

        response = client.get("/api/v1/health")
        assert response.status_code == 200, "Response: " + response.text
        data = response.json()
        assert data["status"] == "no_models_loaded"

    def test_list_models_endpoint(self, client):
        """Test list models endpoint."""
        # Mock service is automatically injected via fixture

        response = client.get("/api/v1/models")
        assert response.status_code == 200
        assert response.json() == []

    def test_predict_endpoint_success(self, client, mock_service):
        """Test successful prediction endpoint."""
        # Reconfigure mock service for this test
        mock_service.list_models.return_value = ["default"]
        mock_service.predict.return_value = ([1], [[0.2, 0.8]], 15.5)

        request_data = {
            "features": [1.0, 2.0, 3.0, 4.0],
            "model_name": "default",
            "return_probabilities": True,
        }

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == [1]
        assert data["probabilities"] == [[0.2, 0.8]]
        assert data["model_name"] == "default"

    def test_predict_endpoint_model_not_found(self, client):
        """Test prediction with non-existent model."""
        # Mock service already configured for this scenario via fixture

        request_data = {"features": [1.0, 2.0, 3.0, 4.0], "model_name": "nonexistent"}

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 404

    def test_load_model_endpoint(self, client, mock_service):
        """Test load model endpoint."""
        # Reconfigure mock service for this test
        mock_service.load_model.return_value = True

        response = client.post("/api/v1/models/test_model/load")
        assert response.status_code == 200
        data = response.json()
        assert "loaded successfully" in data["message"]

    def test_unload_model_endpoint(self, client, mock_service):
        """Test unload model endpoint."""
        # Reconfigure mock service for this test
        mock_service.unload_model.return_value = True

        response = client.delete("/api/v1/models/test_model")
        assert response.status_code == 200
        data = response.json()
        assert "unloaded successfully" in data["message"]

    def test_predict_endpoint_invalid_request(self, client):
        """Test prediction with invalid request."""
        request_data = {
            "features": [],  # Empty features should be invalid
            "model_name": "default",
        }

        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422  # Validation error


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the API."""

    def test_api_startup_and_health(self, client):
        """Test that API starts up and health endpoint works."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "MLOps Template API"
