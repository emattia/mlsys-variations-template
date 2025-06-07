"""Test cases for src/models/training.py."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.training import (
    evaluate_model_cv,
    hyperparameter_tuning,
    load_model,
    save_model,
    train_model,
)


def test_train_model_basic():
    """Test basic model training."""
    X = pl.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
    y = pl.Series([0, 1, 0, 1, 0])

    model = LogisticRegression(random_state=42)

    trained_model = train_model(X, y, model)

    # Check that model is trained
    assert hasattr(trained_model, "coef_")
    assert trained_model.classes_.tolist() == [0, 1]

    # Test prediction
    predictions = trained_model.predict(X.to_numpy())
    assert len(predictions) == 5


def test_train_model_with_numpy():
    """Test training with numpy arrays."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    model = LogisticRegression(random_state=42)

    trained_model = train_model(X, y, model)

    assert hasattr(trained_model, "coef_")
    assert trained_model.classes_.tolist() == [0, 1]


def test_train_model_with_params():
    """Test training with model parameters."""
    X = pl.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [2, 4, 6, 8, 10]})
    y = pl.Series([0, 1, 0, 1, 0])

    model = LogisticRegression()

    trained_model = train_model(X, y, model, random_state=42, max_iter=1000)

    # Check that parameters were set
    assert trained_model.random_state == 42
    assert trained_model.max_iter == 1000


def test_evaluate_model_cv():
    """Test cross-validation evaluation."""
    X = pl.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
    )
    y = pl.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    model = LogisticRegression(random_state=42)

    metrics = evaluate_model_cv(X, y, model, cv=3, scoring="accuracy")

    # Check metrics structure
    assert "mean_cv_score" in metrics
    assert "std_cv_score" in metrics
    assert "min_cv_score" in metrics
    assert "max_cv_score" in metrics
    assert "cv_scores" in metrics
    assert "scoring" in metrics
    assert "cv_folds" in metrics

    # Check values
    assert isinstance(metrics["mean_cv_score"], float)
    assert isinstance(metrics["std_cv_score"], float)
    assert len(metrics["cv_scores"]) == 3
    assert metrics["scoring"] == "accuracy"
    assert metrics["cv_folds"] == 3


def test_evaluate_model_cv_with_params():
    """Test CV evaluation with model parameters."""
    X = np.random.rand(20, 2)
    y = np.random.randint(0, 2, 20)

    model = LogisticRegression()

    metrics = evaluate_model_cv(
        X, y, model, cv=2, scoring="f1", random_state=42, max_iter=1000
    )

    assert metrics["scoring"] == "f1"
    assert metrics["cv_folds"] == 2


def test_hyperparameter_tuning():
    """Test hyperparameter tuning with grid search."""
    X = pl.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        }
    )
    y = pl.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    model = LogisticRegression(random_state=42)
    param_grid = {"C": [0.1, 1.0, 10.0], "max_iter": [100, 1000]}

    best_model, results = hyperparameter_tuning(
        X, y, model, param_grid, cv=2, scoring="accuracy"
    )

    # Check best model
    assert isinstance(best_model, LogisticRegression)
    assert hasattr(best_model, "coef_")

    # Check results structure
    assert "best_params" in results
    assert "best_score" in results
    assert "cv_results" in results

    # Check that best parameters are from the grid
    assert results["best_params"]["C"] in [0.1, 1.0, 10.0]
    assert results["best_params"]["max_iter"] in [100, 1000]

    # Check CV results
    cv_results = results["cv_results"]
    assert "mean_test_score" in cv_results
    assert "std_test_score" in cv_results
    assert "mean_train_score" in cv_results
    assert "std_train_score" in cv_results
    assert "params" in cv_results


def test_hyperparameter_tuning_with_numpy():
    """Test hyperparameter tuning with numpy arrays."""
    X = np.random.rand(15, 3)
    y = np.random.randint(0, 2, 15)

    model = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators": [10, 20], "max_depth": [2, 3]}

    best_model, results = hyperparameter_tuning(X, y, model, param_grid, cv=2)

    assert isinstance(best_model, RandomForestClassifier)
    assert "best_params" in results


def test_save_model():
    """Test saving a trained model."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = LogisticRegression(random_state=42)
    trained_model = train_model(X, y, model)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pkl"
        metadata = {"version": "1.0", "accuracy": 0.85}

        saved_path = save_model(trained_model, model_path, metadata)

        # Check that file was saved
        assert saved_path.exists()
        assert saved_path == model_path

        # Check that we can load the file
        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)

        assert "model" in saved_data
        assert "model_type" in saved_data
        assert "metadata" in saved_data
        assert saved_data["model_type"] == "LogisticRegression"
        assert saved_data["metadata"] == metadata


def test_save_model_no_metadata():
    """Test saving model without metadata."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    model = LogisticRegression(random_state=42)
    trained_model = train_model(X, y, model)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pkl"

        saved_path = save_model(trained_model, model_path)

        assert saved_path.exists()

        with open(model_path, "rb") as f:
            saved_data = pickle.load(f)

        assert saved_data["metadata"] is None


def test_load_model():
    """Test loading a saved model."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = LogisticRegression(random_state=42)
    trained_model = train_model(X, y, model)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_model.pkl"
        metadata = {"version": "1.0", "accuracy": 0.85}

        # Save model
        save_model(trained_model, model_path, metadata)

        # Load model
        loaded_model, loaded_metadata = load_model(model_path)

        # Check that model was loaded correctly
        assert isinstance(loaded_model, LogisticRegression)
        assert loaded_metadata == metadata

        # Check that loaded model makes same predictions
        original_pred = trained_model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)


def test_load_model_file_not_found():
    """Test loading model from non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_model.pkl")


def test_save_model_creates_directory():
    """Test that save_model creates directory if it doesn't exist."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    model = LogisticRegression(random_state=42)
    trained_model = train_model(X, y, model)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a nested path that doesn't exist
        model_path = Path(temp_dir) / "models" / "v1" / "test_model.pkl"

        saved_path = save_model(trained_model, model_path)

        # Check that directory was created and file exists
        assert saved_path.exists()
        assert saved_path.parent.exists()


@patch("src.models.training.logger")
def test_training_logging(mock_logger):
    """Test that training functions log appropriate messages."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = LogisticRegression(random_state=42)

    # Test train_model logging
    train_model(X, y, model)
    mock_logger.info.assert_any_call("Training LogisticRegression model")
    mock_logger.info.assert_any_call("Model training complete")

    # Test evaluate_model_cv logging
    evaluate_model_cv(X, y, model, cv=2)
    mock_logger.info.assert_any_call(
        "Evaluating LogisticRegression model with 2-fold cross-validation"
    )


def test_integration_training_pipeline():
    """Test integration of the complete training pipeline."""
    # Create more realistic dataset
    np.random.seed(42)
    X = np.random.rand(50, 4)
    y = (X[:, 0] + X[:, 1] > X[:, 2] + X[:, 3]).astype(int)

    X_df = pl.DataFrame({f"feature_{i}": X[:, i] for i in range(4)})
    y_series = pl.Series(y)

    # Step 1: Train initial model
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    train_model(X_df, y_series, model)

    # Step 2: Evaluate with cross-validation
    cv_metrics = evaluate_model_cv(X_df, y_series, model, cv=3)

    # Step 3: Hyperparameter tuning
    param_grid = {"n_estimators": [5, 10], "max_depth": [2, 3]}
    best_model, tuning_results = hyperparameter_tuning(
        X_df, y_series, model, param_grid, cv=2
    )

    # Step 4: Save and load best model
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "best_model.pkl"
        metadata = {
            "cv_score": cv_metrics["mean_cv_score"],
            "best_params": tuning_results["best_params"],
        }

        save_model(best_model, model_path, metadata)
        loaded_model, loaded_metadata = load_model(model_path)

        # Verify everything works together
        assert isinstance(loaded_model, RandomForestClassifier)
        assert "cv_score" in loaded_metadata
        assert "best_params" in loaded_metadata

        # Test predictions
        predictions = loaded_model.predict(X)
        assert len(predictions) == 50
        assert all(pred in [0, 1] for pred in predictions)
