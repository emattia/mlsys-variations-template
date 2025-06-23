"""Tests for the workflow modules.

This module contains tests for the data processing and model training workflows.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import pytest

from workflows.batch_inference import run_batch_inference
from workflows.data_ingestion import ingest_data
from workflows.feature_engineering import engineer_features
from workflows.model_evaluation import evaluate_model
from workflows.model_training import train_and_evaluate_model


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature3": ["A", "B", "A", "C", "B"],
            "target": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def sample_data_path(sample_data: pl.DataFrame) -> str:
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        sample_data.write_csv(f.name)
        return f.name


@pytest.fixture
def mock_model() -> mock.MagicMock:
    """Create a mock model for testing."""
    model = mock.MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1, 0])
    model.predict_proba.return_value = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]
    )
    model.feature_importances_ = np.array([0.4, 0.3, 0.3])
    return model


@mock.patch("workflows.data_ingestion.load_data")
@mock.patch("workflows.data_ingestion.save_data")
def test_ingest_data(
    mock_save_data: mock.MagicMock,
    mock_load_data: mock.MagicMock,
    sample_data: pl.DataFrame,
) -> None:
    """Test the data ingestion workflow."""
    # Mock load_data to return sample data
    mock_load_data.return_value = sample_data

    # Mock save_data to return a path
    mock_save_data.return_value = Path("/path/to/output.csv")

    # Run the workflow
    result = ingest_data(
        source_path="/path/to/source.csv",
        destination_path="/path/to/output.csv",
        validate=False,
        generate_report=False,
    )

    # Check that load_data was called
    mock_load_data.assert_called_once_with("/path/to/source.csv")

    # Check that save_data was called
    mock_save_data.assert_called_once()

    # Check result
    assert result["success"] is True
    assert result["rows_count"] == 5
    assert result["columns_count"] == 5


@mock.patch("workflows.feature_engineering.load_data")
@mock.patch("workflows.feature_engineering.save_data")
def test_engineer_features(
    mock_save_data: mock.MagicMock,
    mock_load_data: mock.MagicMock,
    sample_data: pl.DataFrame,
) -> None:
    """Test the feature engineering workflow."""
    # Mock load_data to return sample data
    mock_load_data.return_value = sample_data

    # Mock save_data to return a path
    mock_save_data.return_value = Path("/path/to/output.csv")

    # Run the workflow
    result = engineer_features(
        input_path="/path/to/source.csv",
        output_path="/path/to/output.csv",
        clean=True,
        normalize=True,
        encode=True,
        target_column="target",
    )

    # Check that load_data was called
    mock_load_data.assert_called_once_with("/path/to/source.csv")

    # Check that save_data was called
    mock_save_data.assert_called_once()

    # Check result
    assert result["success"] is True
    assert result["rows_before"] == 5
    assert "numeric_columns" in result
    assert "categorical_columns" in result
    assert result["target_column"] == "target"


@mock.patch("src.plugins.get_plugin")
@mock.patch("workflows.model_training.load_data")
@mock.patch("workflows.model_training.train_model")
@mock.patch("workflows.model_training.evaluate_model_cv")
@mock.patch("workflows.model_training.save_model")
def test_train_and_evaluate_model(
    mock_save_model: mock.MagicMock,
    mock_evaluate_model_cv: mock.MagicMock,
    mock_train_model: mock.MagicMock,
    mock_load_data: mock.MagicMock,
    mock_get_plugin: mock.MagicMock,
    sample_data: pl.DataFrame,
    mock_model: mock.MagicMock,
) -> None:
    """Test the model training workflow."""
    # Mock load_data to return sample data
    mock_load_data.return_value = sample_data

    # Mock train_model to return a model
    mock_train_model.return_value = mock_model

    # Mock evaluate_model_cv to return metrics
    mock_evaluate_model_cv.return_value = {
        "mean_cv_score": 0.8,
        "std_cv_score": 0.1,
        "min_cv_score": 0.7,
        "max_cv_score": 0.9,
        "cv_scores": [0.7, 0.8, 0.9],
        "scoring": "accuracy",
        "cv_folds": 3,
    }

    # Mock save_model to return a path
    mock_save_model.return_value = Path("/path/to/model.pkl")

    # Mock the plugin and its methods
    mock_trainer = mock.MagicMock()
    mock_trainer.initialize.return_value = None

    # Create a mock ComponentResult
    from src.plugins.base import ComponentResult, ComponentStatus

    mock_result = ComponentResult(
        status=ComponentStatus.SUCCESS,
        component_name="sklearn_trainer",
        execution_time=1.5,
        output_data={
            "model_path": "/path/to/model.pkl",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "train_samples": 80,
            "test_samples": 20,
        },
        artifacts={"model": Path("/path/to/model.pkl")},
        metrics={
            "cv_score_mean": 0.8,
            "cv_score_std": 0.1,
            "training_time": 1.5,
        },
        metadata={"model_type": "random_forest", "problem_type": "classification"},
    )
    mock_trainer.execute.return_value = mock_result
    mock_get_plugin.return_value = mock_trainer

    # Run the workflow
    config_overrides = {
        "model": {
            "model_type": "random_forest",
            "problem_type": "classification",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2", "feature3"],
        }
    }
    result = train_and_evaluate_model(
        data_path=Path("/path/to/data.csv"),
        config_overrides=config_overrides,
    )

    # Check that get_plugin was called
    mock_get_plugin.assert_called_once_with("sklearn_trainer")

    # Check that trainer methods were called
    mock_trainer.initialize.assert_called_once()
    mock_trainer.execute.assert_called_once()

    # Check result
    assert result.is_success()
    assert "model_path" in result.output_data
    assert result.metrics["cv_score_mean"] == 0.8
    assert result.metrics["cv_score_std"] == 0.1


@mock.patch("workflows.model_evaluation.load_data")
@mock.patch("workflows.model_evaluation.load_model")
@mock.patch("workflows.model_evaluation.predict")
@mock.patch("workflows.model_evaluation.evaluate_classification_model")
@mock.patch("workflows.model_evaluation.plot_confusion_matrix")
@mock.patch("workflows.model_evaluation.plot_feature_importance")
@mock.patch("workflows.model_evaluation.save_evaluation_results")
def test_evaluate_model(
    mock_save_results: mock.MagicMock,
    mock_plot_importance: mock.MagicMock,
    mock_plot_cm: mock.MagicMock,
    mock_evaluate_model: mock.MagicMock,
    mock_predict: mock.MagicMock,
    mock_load_model: mock.MagicMock,
    mock_load_data: mock.MagicMock,
    sample_data: pl.DataFrame,
    mock_model: mock.MagicMock,
) -> None:
    """Test the model evaluation workflow."""
    # Mock load_data to return sample data
    mock_load_data.return_value = sample_data

    # Mock load_model to return a model and metadata
    mock_load_model.return_value = (
        mock_model,
        {
            "model_type": "random_forest",
            "problem_type": "classification",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "target_column": "target",
        },
    )

    # Mock predict to return predictions
    mock_predict.return_value = np.array([0, 1, 0, 1, 0])

    # Mock evaluate_classification_model to return metrics
    mock_evaluate_model.return_value = {
        "accuracy": 0.8,
        "precision": 0.75,
        "recall": 0.8,
        "f1": 0.77,
        "confusion_matrix": [[2, 1], [0, 2]],
    }

    # Mock plot_confusion_matrix to return a figure
    mock_plot_cm.return_value = mock.MagicMock()

    # Mock plot_feature_importance to return a figure
    mock_plot_importance.return_value = mock.MagicMock()

    # Mock save_evaluation_results to return a path
    mock_save_results.return_value = Path("/path/to/results.json")

    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as output_dir:
        # Run the workflow
        result = evaluate_model(
            model_path="/path/to/model.pkl",
            data_path="/path/to/data.csv",
            output_dir=output_dir,
            generate_plots=True,
        )

        # Check that load_model was called
        mock_load_model.assert_called_once_with("/path/to/model.pkl")

        # Check that load_data was called
        mock_load_data.assert_called_once_with("/path/to/data.csv")

        # Check that predict was called
        mock_predict.assert_called_once()

        # Check that evaluate_classification_model was called
        mock_evaluate_model.assert_called_once()

        # Check that plot_confusion_matrix was called
        mock_plot_cm.assert_called_once()

        # Check that plot_feature_importance was called
        mock_plot_importance.assert_called_once()

        # Check that save_evaluation_results was called
        mock_save_results.assert_called_once()

        # Check result
        assert result["success"] is True
        assert result["model_path"] == "/path/to/model.pkl"
        assert result["data_path"] == "/path/to/data.csv"
        assert result["model_type"] == "random_forest"
        assert result["problem_type"] == "classification"
        assert result["has_target"] is True
        assert "evaluation_results" in result


@mock.patch("workflows.batch_inference.load_model")
@mock.patch("workflows.batch_inference.batch_predict")
def test_run_batch_inference(
    mock_batch_predict: mock.MagicMock,
    mock_load_model: mock.MagicMock,
    mock_model: mock.MagicMock,
) -> None:
    """Test the batch inference workflow."""
    # Mock load_model to return a model and metadata
    mock_load_model.return_value = (
        mock_model,
        {
            "model_type": "random_forest",
            "problem_type": "classification",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "target_column": "target",
        },
    )

    # Mock batch_predict to return a result
    mock_batch_predict.return_value = {
        "model_path": "/path/to/model.pkl",
        "input_path": "/path/to/data.csv",
        "output_path": "/path/to/predictions.csv",
        "num_predictions": 5,
        "feature_columns": ["feature1", "feature2", "feature3"],
        "has_probabilities": True,
        "model_type": "random_forest",
        "model_metadata": {
            "model_type": "random_forest",
            "problem_type": "classification",
        },
    }

    # Run the workflow
    result = run_batch_inference(
        model_path="/path/to/model.pkl",
        input_path="/path/to/data.csv",
        output_path="/path/to/predictions.csv",
        feature_columns=["feature1", "feature2", "feature3"],
        return_probabilities=True,
    )

    # Check that load_model was called
    mock_load_model.assert_called_once_with("/path/to/model.pkl")

    # Check that batch_predict was called
    mock_batch_predict.assert_called_once()

    # Check result
    assert result["success"] is True
    assert result["model_path"] == "/path/to/model.pkl"
    assert result["input_path"] == "/path/to/data.csv"
    assert result["output_path"] == "/path/to/predictions.csv"
    assert result["num_predictions"] == 5
    assert result["has_probabilities"] is True
    assert result["model_type"] == "random_forest"
