"""Model inference functions.

This module contains functions for making predictions with trained models.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from src.ml.training import load_model

logger = logging.getLogger(__name__)


def predict(
    model: BaseEstimator,
    X: pl.DataFrame | np.ndarray,
    return_probabilities: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Make predictions with a trained model.

    Args:
        model: Trained model
        X: Feature matrix
        return_probabilities: Whether to return probability estimates (for classifiers)

    Returns:
        Predictions, and optionally probability estimates
    """
    logger.info(f"Making predictions with {type(model).__name__} model")

    # Convert polars DataFrame to numpy if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()

    # Make predictions
    y_pred = model.predict(X)

    # Return probabilities if requested and available
    if return_probabilities:
        try:
            y_prob = model.predict_proba(X)
            return y_pred, y_prob
        except (AttributeError, NotImplementedError):
            logger.warning("Model does not support probability estimates")
            return y_pred

    return y_pred


def _load_data_file(input_path: Path) -> pl.DataFrame:
    """Load data from a file based on its format."""
    file_format = input_path.suffix.lstrip(".")

    if file_format.lower() == "csv":
        return pl.read_csv(input_path)
    elif file_format.lower() == "parquet":
        return pl.read_parquet(input_path)
    elif file_format.lower() == "json":
        return pl.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def _make_predictions_with_probabilities(
    model, X, return_probabilities: bool
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Make predictions and optionally return probabilities."""
    if return_probabilities:
        try:
            y_pred, y_prob = predict(model, X, return_probabilities=True)
            return y_pred, y_prob, True
        except (AttributeError, NotImplementedError, ValueError, TypeError):
            y_pred = predict(model, X, return_probabilities=False)
            return y_pred, None, False
    else:
        y_pred = predict(model, X, return_probabilities=False)
        return y_pred, None, False


def _create_result_dataframe(
    df: pl.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    id_column: str | None,
    has_probabilities: bool,
) -> pl.DataFrame:
    """Create result DataFrame with predictions and optional probabilities."""
    if id_column is not None:
        result_df = pl.DataFrame({id_column: df[id_column], "prediction": y_pred})
    else:
        result_df = pl.DataFrame({"prediction": y_pred})

    # Add probabilities if available
    if has_probabilities and isinstance(y_prob, np.ndarray):
        if y_prob.shape[1] == 2:  # Binary classification
            result_df = result_df.with_columns(pl.Series("probability", y_prob[:, 1]))
        else:  # Multi-class classification
            for i in range(y_prob.shape[1]):
                result_df = result_df.with_columns(
                    pl.Series(f"probability_class_{i}", y_prob[:, i])
                )

    return result_df


def _save_results(result_df: pl.DataFrame, output_path: Path) -> None:
    """Save results to file based on output format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_path.suffix.lstrip(".")

    if output_format.lower() == "csv":
        result_df.write_csv(output_path)
    elif output_format.lower() == "parquet":
        result_df.write_parquet(output_path)
    elif output_format.lower() == "json":
        result_df.write_json(output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def batch_predict(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    feature_columns: list[str] | None = None,
    id_column: str | None = None,
    return_probabilities: bool = False,
) -> dict[str, Any]:
    """Make batch predictions with a trained model.

    Args:
        model_path: Path to the saved model
        input_path: Path to the input data file
        output_path: Path where the predictions will be saved
        feature_columns: List of feature columns to use (None for all columns)
        id_column: Name of the ID column (None if no ID column)
        return_probabilities: Whether to return probability estimates (for classifiers)

    Returns:
        Dictionary with information about the batch prediction
    """
    logger.info(f"Starting batch prediction: {input_path} -> {output_path}")

    # Load model and data
    model, metadata = load_model(model_path)
    df = _load_data_file(Path(input_path))

    # Extract features
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != id_column]
    X = df.select(feature_columns)

    # Make predictions
    y_pred, y_prob, has_probabilities = _make_predictions_with_probabilities(
        model, X, return_probabilities
    )

    # Create and save results
    result_df = _create_result_dataframe(
        df, y_pred, y_prob, id_column, has_probabilities
    )
    _save_results(result_df, Path(output_path))

    # Return information about the batch prediction
    return {
        "model_path": str(model_path),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_predictions": len(result_df),
        "feature_columns": feature_columns,
        "has_probabilities": has_probabilities,
        "model_type": type(model).__name__,
        "model_metadata": metadata,
    }


def _convert_dataframe_to_payload(X: pl.DataFrame) -> dict[str, Any]:
    """Convert DataFrame to API payload."""
    data = {col: X[col].to_list() for col in X.columns}
    return {"data": data, "feature_names": list(X.columns)}


def _convert_dict_to_payload(X: dict[str, list[float]]) -> dict[str, Any]:
    """Convert dictionary to API payload."""
    if not all(isinstance(v, list) for v in X.values()):
        raise ValueError("All values in the dictionary must be lists")
    return {"data": X, "feature_names": list(X.keys())}


def _convert_list_to_payload(
    X: list[list[float]], feature_names: list[str]
) -> dict[str, Any]:
    """Convert list of lists to API payload."""
    if feature_names is None:
        raise ValueError("feature_names is required when X is a list of lists")

    if len(X[0]) != len(feature_names):
        raise ValueError(
            "Length of feature_names must match the number of features in X"
        )

    data = {feature: [x[i] for x in X] for i, feature in enumerate(feature_names)}
    return {"data": data, "feature_names": feature_names}


def create_prediction_payload(
    X: pl.DataFrame | dict[str, list[float]] | list[list[float]],
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Create a payload for prediction API requests.

    Args:
        X: Feature data (DataFrame, dictionary, or list of lists)
        feature_names: List of feature names (required if X is a list of lists)

    Returns:
        Dictionary payload for API request
    """
    if isinstance(X, pl.DataFrame):
        return _convert_dataframe_to_payload(X)
    elif isinstance(X, dict):
        return _convert_dict_to_payload(X)
    elif isinstance(X, list) and all(isinstance(x, list) for x in X):
        return _convert_list_to_payload(X, feature_names)
    else:
        raise ValueError("X must be a DataFrame, dictionary, or list of lists")


def parse_prediction_response(
    response: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse a prediction API response.

    Args:
        response: Dictionary response from prediction API

    Returns:
        Tuple of (predictions, probabilities)
    """
    # Extract predictions
    if "predictions" not in response:
        raise ValueError("Response does not contain 'predictions'")

    predictions = np.array(response["predictions"])

    # Extract probabilities if available
    probabilities = None
    if "probabilities" in response:
        probabilities = np.array(response["probabilities"])

    return predictions, probabilities
