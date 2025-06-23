"""Test cases for src/models/inference.py."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from unittest.mock import Mock

from src.models.inference import (
    batch_predict,
    predict,
)


def test_predict_basic():
    """Test basic prediction functionality."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    predictions = predict(model, X)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3
    assert all(pred in [0, 1] for pred in predictions)


def test_predict_with_polars():
    """Test prediction with Polars DataFrame input."""
    X_df = pl.DataFrame({"feature1": [1, 2, 3], "feature2": [2, 4, 6]})
    y = np.array([0, 1, 0])
    model = LogisticRegression(random_state=42)
    model.fit(X_df.to_numpy(), y)
    predictions = predict(model, X_df)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 3


def test_predict_with_probabilities():
    """Test prediction with probability estimates."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X, y)
    predictions, probabilities = predict(model, X, return_probabilities=True)
    assert isinstance(predictions, np.ndarray)
    assert isinstance(probabilities, np.ndarray)
    assert len(predictions) == 4
    assert probabilities.shape == (4, 2)  # Binary classification
    assert all(0 <= prob <= 1 for row in probabilities for prob in row)


def test_predict_no_probabilities_support():
    """Test prediction when model doesn't support probabilities."""
    X = np.array([[1, 2], [3, 4]])

    # Mock model without predict_proba
    model = Mock()
    model.predict.return_value = np.array([0, 1])
    model.predict_proba.side_effect = AttributeError("No probabilities")
    result = predict(model, X, return_probabilities=True)

    # Should return only predictions when probabilities aren't available
    assert isinstance(result, np.ndarray)
    assert not isinstance(result, tuple)


def test_batch_predict_csv():
    """Test batch prediction with CSV files."""
    # Create test data
    test_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": [2.0, 4.0, 6.0, 8.0],
        }
    )

    # Train a simple model
    X = test_df.select(["feature1", "feature2"]).to_numpy()
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = Path(temp_dir) / "model.pkl"
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "model_type": "LogisticRegression",
                    "metadata": {"version": "1.0"},
                },
                f,
            )

        # Save input data
        input_path = Path(temp_dir) / "input.csv"
        test_df.write_csv(input_path)

        # Define output path
        output_path = Path(temp_dir) / "predictions.csv"

        # Run batch prediction
        result = batch_predict(
            model_path=model_path,
            input_path=input_path,
            output_path=output_path,
            feature_columns=["feature1", "feature2"],
            id_column="id",
        )

        # Check result metadata
        assert result["model_path"] == str(model_path)
        assert result["input_path"] == str(input_path)
        assert result["output_path"] == str(output_path)
        assert result["num_predictions"] == 4
        assert result["feature_columns"] == ["feature1", "feature2"]
        assert result["model_type"] == "LogisticRegression"

        # Check output file
        assert output_path.exists()
        predictions_df = pl.read_csv(output_path)
        assert "id" in predictions_df.columns
        assert "prediction" in predictions_df.columns
        assert len(predictions_df) == 4


def test_batch_predict_with_probabilities():
    """Test batch prediction with probabilities."""
    test_df = pl.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [2.0, 4.0, 6.0]})

    # Train model that supports probabilities
    X = test_df.to_numpy()
    y = np.array([0, 1, 0])
    model = RandomForestClassifier(random_state=42, n_estimators=5)
    model.fit(X, y)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = Path(temp_dir) / "model.pkl"
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "model_type": "RandomForestClassifier",
                    "metadata": None,
                },
                f,
            )

        # Save input data
        input_path = Path(temp_dir) / "input.csv"
        test_df.write_csv(input_path)

        # Define output path
        output_path = Path(temp_dir) / "predictions.csv"

        # Run batch prediction with probabilities
        result = batch_predict(
            model_path=model_path,
            input_path=input_path,
            output_path=output_path,
            return_probabilities=True,
        )
        assert result["has_probabilities"] is True

        # Check output file has probability column
        predictions_df = pl.read_csv(output_path)
        assert "prediction" in predictions_df.columns
        assert "probability" in predictions_df.columns  # Binary classification


def test_batch_predict_parquet():
    """Test batch prediction with Parquet files."""
    test_df = pl.DataFrame({"feature1": [1.0, 2.0], "feature2": [2.0, 4.0]})
    X = test_df.to_numpy()
    y = np.array([0, 1])
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = Path(temp_dir) / "model.pkl"
        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(
                {"model": model, "model_type": "LogisticRegression", "metadata": {}}, f
            )

        # Save input as Parquet
        input_path = Path(temp_dir) / "input.parquet"
        test_df.write_parquet(input_path)

        # Define output path
        output_path = Path(temp_dir) / "predictions.parquet"

        # Run batch prediction
        result = batch_predict(
            model_path=model_path, input_path=input_path, output_path=output_path
        )
        assert result["num_predictions"] == 2
        assert output_path.exists()


def test_batch_predict_unsupported_format():
    """Test batch prediction with unsupported file format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy files
        model_path = Path(temp_dir) / "model.pkl"
        input_path = Path(temp_dir) / "input.txt"  # Unsupported format
        output_path = Path(temp_dir) / "output.csv"

        # Create a dummy model file
        import pickle

        model = LogisticRegression()
        with open(model_path, "wb") as f:
            pickle.dump(
                {"model": model, "model_type": "LogisticRegression", "metadata": None},
                f,
            )

        # Create dummy input file
        input_path.write_text("not a valid data file")

        # Should raise error for unsupported format
        with pytest.raises(ValueError, match="Unsupported file format"):
            batch_predict(
                model_path=model_path,
                input_path=input_path,
                output_path=output_path,
            )


def test_batch_predict_feature_selection():
    """Test batch prediction with specific feature column selection."""
    test_df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [2.0, 4.0, 6.0],
            "extra_column": ["a", "b", "c"],  # Should be ignored
        }
    )

    # Train model with only feature1 and feature2
    X = test_df.select(["feature1", "feature2"]).to_numpy()
    y = np.array([0, 1, 0])
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = Path(temp_dir) / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {"model": model, "model_type": "LogisticRegression", "metadata": {}}, f
            )

        # Save input data
        input_path = Path(temp_dir) / "input.csv"
        test_df.write_csv(input_path)

        # Define output path
        output_path = Path(temp_dir) / "predictions.csv"

        # Run batch prediction with specific feature columns
        result = batch_predict(
            model_path=model_path,
            input_path=input_path,
            output_path=output_path,
            feature_columns=["feature1", "feature2"],
            id_column="id",
        )

        # Should work correctly with selected features
        assert result["num_predictions"] == 3
        assert result["feature_columns"] == ["feature1", "feature2"]

        # Check output
        predictions_df = pl.read_csv(output_path)
        assert "id" in predictions_df.columns
        assert "prediction" in predictions_df.columns
        assert len(predictions_df) == 3


def test_batch_predict_model_loading_error():
    """Test batch prediction with model loading error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a corrupted model file
        model_path = Path(temp_dir) / "corrupted_model.pkl"
        model_path.write_text("This is not a valid pickle file")

        # Create dummy input file
        input_path = Path(temp_dir) / "input.csv"
        test_df = pl.DataFrame({"feature": [1, 2, 3]})
        test_df.write_csv(input_path)

        # Define output path
        output_path = Path(temp_dir) / "output.csv"

        # Should raise error when trying to load corrupted model
        with pytest.raises((pickle.PickleError, ValueError, EOFError)):
            batch_predict(
                model_path=model_path,
                input_path=input_path,
                output_path=output_path,
            )


def test_batch_predict_data_validation():
    """Test batch prediction with data validation errors."""
    # Create a simple trained model
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = Path(temp_dir) / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {"model": model, "model_type": "LogisticRegression", "metadata": {}}, f
            )

        # Create input data with wrong number of features
        wrong_features_df = pl.DataFrame({"feature1": [1, 2, 3]})  # Missing feature2
        input_path = Path(temp_dir) / "input.csv"
        wrong_features_df.write_csv(input_path)

        # Define output path
        output_path = Path(temp_dir) / "output.csv"

        # Should handle feature mismatch gracefully
        with pytest.raises((ValueError, Exception)):
            batch_predict(
                model_path=model_path,
                input_path=input_path,
                output_path=output_path,
                feature_columns=[
                    "feature1"
                ],  # Model expects 2 features but only 1 provided
            )
