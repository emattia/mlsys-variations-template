"""Test cases for src/models/inference.py."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.inference import (
    batch_predict,
    create_prediction_payload,
    parse_prediction_response,
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
        input_path.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            batch_predict(
                model_path=model_path, input_path=input_path, output_path=output_path
            )


def test_create_prediction_payload_dataframe():
    """Test creating payload from Polars DataFrame."""
    X_df = pl.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [2.0, 4.0, 6.0]})

    payload = create_prediction_payload(X_df)

    assert "data" in payload
    assert "feature_names" in payload
    assert payload["feature_names"] == ["feature1", "feature2"]
    assert payload["data"]["feature1"] == [1.0, 2.0, 3.0]
    assert payload["data"]["feature2"] == [2.0, 4.0, 6.0]


def test_create_prediction_payload_dict():
    """Test creating payload from dictionary."""
    X_dict = {"feature1": [1.0, 2.0, 3.0], "feature2": [2.0, 4.0, 6.0]}

    payload = create_prediction_payload(X_dict)

    assert payload["data"] == X_dict
    assert payload["feature_names"] == ["feature1", "feature2"]


def test_create_prediction_payload_list_of_lists():
    """Test creating payload from list of lists."""
    X_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    feature_names = ["feature1", "feature2"]

    payload = create_prediction_payload(X_list, feature_names)

    assert payload["data"]["feature1"] == [1.0, 3.0, 5.0]
    assert payload["data"]["feature2"] == [2.0, 4.0, 6.0]
    assert payload["feature_names"] == feature_names


def test_create_prediction_payload_list_no_feature_names():
    """Test error when creating payload from list without feature names."""
    X_list = [[1.0, 2.0], [3.0, 4.0]]

    with pytest.raises(ValueError, match="feature_names is required"):
        create_prediction_payload(X_list)


def test_create_prediction_payload_invalid_dict():
    """Test error with invalid dictionary format."""
    X_dict = {
        "feature1": [1.0, 2.0],
        "feature2": "not a list",  # Invalid value
    }

    with pytest.raises(ValueError, match="All values in the dictionary must be lists"):
        create_prediction_payload(X_dict)


def test_create_prediction_payload_mismatched_lengths():
    """Test error with mismatched feature lengths."""
    X_list = [[1.0, 2.0], [3.0, 4.0]]
    feature_names = ["feature1"]  # Wrong length

    with pytest.raises(ValueError, match="Length of feature_names must match"):
        create_prediction_payload(X_list, feature_names)


def test_create_prediction_payload_invalid_type():
    """Test error with invalid input type."""
    with pytest.raises(
        ValueError, match="X must be a DataFrame, dictionary, or list of lists"
    ):
        create_prediction_payload("invalid input")


def test_parse_prediction_response():
    """Test parsing prediction response."""
    response = {
        "predictions": [0, 1, 0, 1],
        "probabilities": [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]],
    }

    predictions, probabilities = parse_prediction_response(response)

    assert isinstance(predictions, np.ndarray)
    assert isinstance(probabilities, np.ndarray)
    assert predictions.tolist() == [0, 1, 0, 1]
    assert probabilities.shape == (4, 2)


def test_parse_prediction_response_no_probabilities():
    """Test parsing response without probabilities."""
    response = {"predictions": [0, 1, 0]}

    predictions, probabilities = parse_prediction_response(response)

    assert isinstance(predictions, np.ndarray)
    assert probabilities is None
    assert predictions.tolist() == [0, 1, 0]


def test_parse_prediction_response_invalid():
    """Test error with invalid response format."""
    response = {"invalid_key": [0, 1, 0]}

    with pytest.raises(ValueError, match="Response does not contain 'predictions'"):
        parse_prediction_response(response)


@patch("src.models.inference.logger")
def test_inference_logging(mock_logger):
    """Test that inference functions log appropriate messages."""
    X = np.array([[1, 2], [3, 4]])
    model = LogisticRegression(random_state=42)
    model.fit(X, [0, 1])

    predict(model, X)

    mock_logger.info.assert_called_with(
        "Making predictions with LogisticRegression model"
    )


def test_integration_inference_pipeline():
    """Test integration of the complete inference pipeline."""
    # Create training data
    np.random.seed(42)
    X_train = np.random.rand(50, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > X_train[:, 2] + X_train[:, 3]).astype(
        int
    )

    # Train model
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X_train, y_train)

    # Create test data
    X_test = np.random.rand(10, 4)
    X_test_df = pl.DataFrame({f"feature_{i}": X_test[:, i] for i in range(4)})

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = Path(temp_dir) / "model.pkl"
        import pickle

        metadata = {"accuracy": 0.85, "features": 4}
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "model_type": "RandomForestClassifier",
                    "metadata": metadata,
                },
                f,
            )

        # Test direct prediction
        predictions = predict(model, X_test)
        assert len(predictions) == 10

        # Test prediction with probabilities
        predictions_with_prob, probabilities = predict(
            model, X_test, return_probabilities=True
        )
        assert np.array_equal(predictions, predictions_with_prob)
        assert probabilities.shape == (10, 2)

        # Test batch prediction
        input_path = Path(temp_dir) / "test_data.csv"
        X_test_df.write_csv(input_path)

        output_path = Path(temp_dir) / "batch_predictions.csv"

        batch_result = batch_predict(
            model_path=model_path,
            input_path=input_path,
            output_path=output_path,
            return_probabilities=True,
        )

        # Verify batch prediction results
        assert batch_result["num_predictions"] == 10
        assert batch_result["has_probabilities"] is True
        assert batch_result["model_metadata"] == metadata

        # Load and verify predictions
        predictions_df = pl.read_csv(output_path)
        assert len(predictions_df) == 10
        assert "prediction" in predictions_df.columns
        assert "probability" in predictions_df.columns

        # Test prediction payload creation
        payload = create_prediction_payload(X_test_df)
        assert "data" in payload
        assert "feature_names" in payload
        assert len(payload["feature_names"]) == 4

        # Test response parsing
        mock_response = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }
        parsed_preds, parsed_probs = parse_prediction_response(mock_response)
        assert np.array_equal(parsed_preds, predictions)
        assert np.allclose(parsed_probs, probabilities)
