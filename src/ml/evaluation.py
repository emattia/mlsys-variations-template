"""Model evaluation functions.

This module contains functions for evaluating machine learning models.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_classification_model(
    model: BaseEstimator,
    X: pl.DataFrame | np.ndarray,
    y: pl.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a classification model.

    Args:
        model: Trained classification model
        X: Feature matrix
        y: True labels
        threshold: Probability threshold for binary classification

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating classification model")

    # Convert polars DataFrame/Series to numpy if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()

    # Get predictions with proper error handling
    try:
        y_pred = model.predict(X)
    except Exception as e:
        # Re-raise as RuntimeError for consistent error handling
        raise RuntimeError(f"Model prediction failed: {e}") from e

    # Get probabilities if available
    try:
        y_prob = model.predict_proba(X)
        has_probabilities = True
    except (AttributeError, NotImplementedError):
        has_probabilities = False

    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="weighted")),
        "recall": float(recall_score(y, y_pred, average="weighted")),
        "f1": float(f1_score(y, y_pred, average="weighted")),
    }

    # Add ROC AUC if probabilities are available and it's binary classification
    if has_probabilities and len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob[:, 1]))

    # Calculate confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    logger.info(
        f"Classification metrics: accuracy={metrics['accuracy']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )
    return metrics


def evaluate_regression_model(
    model: BaseEstimator,
    X: pl.DataFrame | np.ndarray,
    y: pl.Series | np.ndarray,
) -> dict[str, float]:
    """Evaluate a regression model.

    Args:
        model: Trained regression model
        X: Feature matrix
        y: True values

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating regression model")

    # Convert polars DataFrame/Series to numpy if needed
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pl.Series):
        y = y.to_numpy()

    # Get predictions
    y_pred = model.predict(X)

    # Calculate metrics
    metrics = {
        "mse": float(mean_squared_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }

    logger.info(
        f"Regression metrics: rmse={metrics['rmse']:.4f}, r2={metrics['r2']:.4f}"
    )
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        figsize: Figure size
        cmap: Colormap
        output_path: Path to save the plot (if None, the plot is not saved)

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    # Save plot if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Confusion matrix plot saved to {output_path}")

    return fig


def plot_feature_importance(
    feature_names: list[str],
    importance_values: list[float],
    top_n: int | None = None,
    figsize: tuple[int, int] = (12, 8),
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot feature importance.

    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        top_n: Number of top features to plot (None for all)
        figsize: Figure size
        output_path: Path to save the plot (if None, the plot is not saved)

    Returns:
        Matplotlib figure
    """
    # Create dictionary of feature importances
    importance_dict = dict(zip(feature_names, importance_values, strict=True))

    # Sort by importance (descending)
    sorted_importance = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    # Get top N if specified
    if top_n is not None:
        sorted_importance = dict(list(sorted_importance.items())[:top_n])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot feature importance
    ax.barh(list(sorted_importance.keys()), list(sorted_importance.values()))

    # Set labels
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance")

    fig.tight_layout()

    # Save plot if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")

    return fig


def save_evaluation_results(results: dict[str, Any], output_path: str | Path) -> Path:
    """Save evaluation results to a JSON file.

    Args:
        results: Dictionary with evaluation results
        output_path: Path where the results will be saved

    Returns:
        Path to the saved results file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists
    def convert_numpy(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, np.int64 | np.int32 | np.float64 | np.float32):
            return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
        else:
            return obj

    results_json = convert_numpy(results)

    # Save results
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"Evaluation results saved to {output_path}")
    return output_path
