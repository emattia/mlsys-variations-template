"""Tests for the API and batch endpoints.

This module contains tests for the prediction API and batch processor endpoints.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from endpoints.api.prediction import app
from endpoints.batch.batch_processor import BatchRequestHandler


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model() -> mock.MagicMock:
    """Create a mock model for testing."""
    model = mock.MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1])
    model.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]]
    )
    return model


@pytest.fixture
def mock_metadata() -> dict[str, Any]:
    """Create mock model metadata for testing."""
    return {
        "model_type": "random_forest",
        "problem_type": "classification",
        "feature_columns": ["feature1", "feature2", "feature3"],
        "target_column": "target",
    }


def test_root_endpoint(test_client: TestClient) -> None:
    """Test the root endpoint."""
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Model Prediction API"}


@mock.patch("endpoints.api.prediction.get_model")
def test_predict_endpoint(
    mock_get_model: mock.MagicMock,
    test_client: TestClient,
    mock_model: mock.MagicMock,
    mock_metadata: dict[str, Any],
) -> None:
    """Test the predict endpoint."""
    # Mock the get_model function
    mock_get_model.return_value = (mock_model, mock_metadata)

    # Create test data
    data = {
        "data": {
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": [5.0, 6.0, 7.0, 8.0],
            "feature3": [9.0, 10.0, 11.0, 12.0],
        },
        "return_probabilities": True,
    }

    # Make request
    response = test_client.post("/predict", json=data)

    # Check response
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert "probabilities" in result
    assert "model_info" in result
    assert len(result["predictions"]) == 4
    assert len(result["probabilities"]) == 4
    assert result["model_info"]["type"] == "random_forest"
    assert result["model_info"]["problem_type"] == "classification"


@mock.patch("endpoints.api.prediction.get_model")
def test_predict_endpoint_without_probabilities(
    mock_get_model: mock.MagicMock,
    test_client: TestClient,
    mock_model: mock.MagicMock,
    mock_metadata: dict[str, Any],
) -> None:
    """Test the predict endpoint without returning probabilities."""
    # Mock the get_model function
    mock_get_model.return_value = (mock_model, mock_metadata)

    # Create test data
    data = {
        "data": {
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": [5.0, 6.0, 7.0, 8.0],
            "feature3": [9.0, 10.0, 11.0, 12.0],
        },
        "return_probabilities": False,
    }

    # Make request
    response = test_client.post("/predict", json=data)

    # Check response
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert result["probabilities"] is None
    assert "model_info" in result
    assert len(result["predictions"]) == 4
    assert result["model_info"]["type"] == "random_forest"
    assert result["model_info"]["problem_type"] == "classification"


@mock.patch("endpoints.batch.batch_processor.batch_predict")
def test_batch_request_handler(
    mock_batch_predict: mock.MagicMock, sample_data_path: str
) -> None:
    """Test the batch request handler."""
    # Create temporary directories
    with tempfile.TemporaryDirectory() as input_dir_str:
        with tempfile.TemporaryDirectory() as output_dir_str:
            with tempfile.TemporaryDirectory() as model_dir_str:
                input_dir = Path(input_dir_str)
                output_dir = Path(output_dir_str)
                model_dir = Path(model_dir_str)

                # Mock batch_predict
                mock_batch_predict.return_value = {
                    "num_predictions": 10,
                    "has_probabilities": True,
                    "model_path": str(model_dir / "model.pkl"),
                    "input_path": "_data.csv",
                    "output_path": str(output_dir / "predictions.csv"),
                }

                # Create a request file
                request_data = {
                    "data_path": "_data.csv",
                    "model_name": "model",
                    "return_probabilities": True,
                }
                request_path = input_dir / "request.json"
                with open(request_path, "w") as f:
                    json.dump(request_data, f)

                # Create a mock model file
                model_path = model_dir / "model.pkl"
                model_path.touch()

                # Create handler
                handler = BatchRequestHandler(input_dir, output_dir, model_dir)

                # Process the request file
                handler._process_request_file(request_path)

                # Check that batch_predict was called
                mock_batch_predict.assert_called_once()

                # Check that the request file was moved to processed directory
                assert not request_path.exists()
                assert (input_dir / "processed" / "request.json").exists()

                # Check that a response file was created
                response_files = list(output_dir.glob("*_response.json"))
                assert len(response_files) == 1

                # Check response content
                with open(response_files[0]) as f:
                    response = json.load(f)
                assert response["status"] == "success"
                assert response["num_predictions"] == 10
                assert response["has_probabilities"] is True


@mock.patch("endpoints.batch.batch_processor.batch_predict")
def test_batch_request_handler_error(
    mock_batch_predict: mock.MagicMock, sample_data_path: str
) -> None:
    """Test the batch request handler with an error."""
    # Create temporary directories
    with tempfile.TemporaryDirectory() as input_dir_str:
        with tempfile.TemporaryDirectory() as output_dir_str:
            with tempfile.TemporaryDirectory() as model_dir_str:
                input_dir = Path(input_dir_str)
                output_dir = Path(output_dir_str)
                model_dir = Path(model_dir_str)

                # Mock batch_predict to raise an exception
                mock_batch_predict.side_effect = ValueError("Test error")

                # Create a request file
                request_data = {
                    "data_path": "_data.csv",
                    "model_name": "model",
                    "return_probabilities": True,
                }
                request_path = input_dir / "request.json"
                with open(request_path, "w") as f:
                    json.dump(request_data, f)

                # Create a mock model file
                model_path = model_dir / "model.pkl"
                model_path.touch()

                # Create handler
                handler = BatchRequestHandler(input_dir, output_dir, model_dir)

                # Process the request file
                handler._process_request_file(request_path)

                # Check that batch_predict was called
                mock_batch_predict.assert_called_once()

                # Check that the request file was moved to failed directory
                assert not request_path.exists()
                assert (input_dir / "failed" / "request.json").exists()

                # Check that an error file was created
                error_files = list(output_dir.glob("*_error.json"))
                assert len(error_files) == 1

                # Check error content
                with open(error_files[0]) as f:
                    error = json.load(f)
                assert error["status"] == "error"
                assert "Test error" in error["error"]
