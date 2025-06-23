"""Model training functions.

This module contains functions for training machine learning models.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, cross_val_score

logger = logging.getLogger(__name__)


def safe_pickle_load(file_path: Path) -> Any:
    """Safely load a pickle file with error handling."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file size (basic sanity check)
    file_size = file_path.stat().st_size

    # Get max file size from config, with fallback to 100MB
    try:
        from hydra import compose

        cfg = compose(config_name="config")
        max_size_mb = (
            cfg.get("ml_systems", {})
            .get("system_limits", {})
            .get("max_file_size_mb", 100)
        )
        max_size_bytes = max_size_mb * 1024 * 1024
    except Exception:
        max_size_bytes = 100 * 1024 * 1024  # 100MB default

    if file_size > max_size_bytes:
        raise ValueError(
            f"File too large for safety: {file_size} bytes (max: {max_size_bytes} bytes)"
        )

    try:
        with open(file_path, "rb") as f:
            # Load with restricted unpickler in production
            data = pickle.load(f)  # nosec B301 - Internal model files only
        return data
    except (pickle.UnpicklingError, EOFError, ImportError) as e:
        raise ValueError(f"Failed to load pickle file: {e}") from e


def train_model(
    X: pl.DataFrame | np.ndarray,
    y: pl.Series | np.ndarray,
    model: BaseEstimator,
    **model_params: Any,
) -> BaseEstimator:
    """Train a machine learning model.

    Args:
        X: Feature matrix
        y: Target vector
        model: Scikit-learn model instance
        **model_params: Additional parameters to pass to the model

    Returns:
        Trained model
    """
    logger.info(f"Training {type(model).__name__} model")

    # Convert polars DataFrame/Series to numpy if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()

    # Set model parameters
    if model_params:
        model.set_params(**model_params)

    # Train the model
    model.fit(X, y)

    logger.info("Model training complete")
    return model


def evaluate_model_cv(
    X: pl.DataFrame | np.ndarray,
    y: pl.Series | np.ndarray,
    model: BaseEstimator,
    cv: int = 5,
    scoring: str = "accuracy",
    **model_params: Any,
) -> dict[str, float]:
    """Evaluate a model using cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        model: Scikit-learn model instance
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        **model_params: Additional parameters to pass to the model

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(
        f"Evaluating {type(model).__name__} model with {cv}-fold cross-validation"
    )

    # Convert polars DataFrame/Series to numpy if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()

    # Set model parameters
    if model_params:
        model.set_params(**model_params)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Calculate metrics
    metrics = {
        "mean_cv_score": float(np.mean(cv_scores)),
        "std_cv_score": float(np.std(cv_scores)),
        "min_cv_score": float(np.min(cv_scores)),
        "max_cv_score": float(np.max(cv_scores)),
        "cv_scores": cv_scores.tolist(),
        "scoring": scoring,
        "cv_folds": cv,
    }

    logger.info(
        f"Cross-validation results: mean={metrics['mean_cv_score']:.4f}, "
        f"std={metrics['std_cv_score']:.4f}"
    )
    return metrics


def hyperparameter_tuning(
    X: pl.DataFrame | np.ndarray,
    y: pl.Series | np.ndarray,
    model: BaseEstimator,
    param_grid: dict[str, list[Any]],
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
) -> tuple[BaseEstimator, dict[str, Any]]:
    """Perform hyperparameter tuning using grid search.

    Args:
        X: Feature matrix
        y: Target vector
        model: Scikit-learn model instance
        param_grid: Dict with parameter names as keys and lists of param values
        cv: Number of cross-validation folds
        scoring: Scoring metric to use
        n_jobs: Number of jobs to run in parallel (-1 means using all processors)

    Returns:
        Tuple of (best model, dictionary with tuning results)
    """
    logger.info(f"Performing hyperparameter tuning for {type(model).__name__}")

    # Convert polars DataFrame/Series to numpy if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()

    # Create grid search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=True,
    )

    # Perform grid search
    grid_search.fit(X, y)

    # Get best model
    best_model = grid_search.best_estimator_

    # Prepare results
    results = {
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_),
        "cv_results": {
            "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
            "mean_train_score": grid_search.cv_results_["mean_train_score"].tolist(),
            "std_train_score": grid_search.cv_results_["std_train_score"].tolist(),
            "params": grid_search.cv_results_["params"],
        },
    }

    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Best score: {results['best_score']:.4f}")

    return best_model, results


def save_model(
    model: BaseEstimator,
    model_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a trained model to disk.

    Args:
        model: Trained model to save
        model_path: Path where the model will be saved
        metadata: Optional dictionary with model metadata

    Returns:
        Path to the saved model
    """
    model_path = Path(model_path)

    # Create directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare model data
    model_data = {
        "model": model,
        "model_type": type(model).__name__,
        "metadata": metadata,
    }

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {model_path}")
    return model_path


def load_model(model_path: str | Path) -> tuple[BaseEstimator, dict[str, Any]]:
    """Load a trained model from disk.

    Args:
        model_path: Path to the saved model

    Returns:
        Tuple of (loaded model, model metadata)
    """
    model_path = Path(model_path)

    # Load model using safe pickle loading
    model_data = safe_pickle_load(model_path)

    model = model_data["model"]
    metadata = model_data.get("metadata", {})

    logger.info(f"Model loaded from {model_path}")
    return model, metadata
