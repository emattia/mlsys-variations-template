"""Test cases for src/models/evaluation.py."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.models.evaluation import (
    evaluate_classification_model,
    evaluate_regression_model,
    plot_confusion_matrix,
    plot_feature_importance,
    save_evaluation_results,
)


def test_evaluate_classification_model_basic():
    """Test basic classification model evaluation."""
    # Create simple dataset
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 1, 0, 1, 0, 1])

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Evaluate
    metrics = evaluate_classification_model(model, X, y)

    # Check metrics structure
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics

    # Check metric types
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["precision"], float)
    assert isinstance(metrics["recall"], float)
    assert isinstance(metrics["f1"], float)
    assert isinstance(metrics["confusion_matrix"], list)

    # Check metric ranges (should be between 0 and 1)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_evaluate_classification_model_with_probabilities():
    """Test classification evaluation with probability estimates."""
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)

    # Train model that supports probabilities
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X, y)

    metrics = evaluate_classification_model(model, X, y)

    # Should include ROC AUC for binary classification with probabilities
    assert "roc_auc" in metrics
    assert isinstance(metrics["roc_auc"], float)
    assert 0 <= metrics["roc_auc"] <= 1


def test_evaluate_classification_model_polars_input():
    """Test classification evaluation with Polars DataFrame input."""
    X_df = pl.DataFrame(
        {"feature1": [1, 2, 3, 4, 5, 6], "feature2": [2, 4, 6, 8, 10, 12]}
    )
    y_series = pl.Series([0, 1, 0, 1, 0, 1])

    model = LogisticRegression(random_state=42)
    model.fit(X_df.to_numpy(), y_series.to_numpy())

    metrics = evaluate_classification_model(model, X_df, y_series)

    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)


def test_evaluate_classification_model_multiclass():
    """Test multiclass classification evaluation."""
    X = np.random.rand(30, 4)
    y = np.random.randint(0, 3, 30)  # 3 classes

    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(X, y)

    metrics = evaluate_classification_model(model, X, y)

    # Should not include ROC AUC for multiclass
    assert "roc_auc" not in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_evaluate_classification_model_no_probabilities():
    """Test classification evaluation with model that doesn't support probabilities."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    # Create a mock model without predict_proba
    model = Mock()
    model.predict.return_value = np.array([0, 1, 0, 1])
    model.predict_proba.side_effect = AttributeError("No probabilities")

    metrics = evaluate_classification_model(model, X, y)

    # Should not include ROC AUC
    assert "roc_auc" not in metrics
    assert "accuracy" in metrics


def test_evaluate_regression_model_basic():
    """Test basic regression model evaluation."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # y = 2*x

    model = LinearRegression()
    model.fit(X, y)

    metrics = evaluate_regression_model(model, X, y)

    # Check metrics structure
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    # Check metric types
    assert isinstance(metrics["mse"], float)
    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["r2"], float)

    # For perfect linear relationship, R² should be close to 1
    assert metrics["r2"] > 0.99

    # RMSE should be sqrt of MSE
    assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-10


def test_evaluate_regression_model_polars_input():
    """Test regression evaluation with Polars DataFrame input."""
    X_df = pl.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y_series = pl.Series([2, 4, 6, 8, 10])

    model = LinearRegression()
    model.fit(X_df.to_numpy(), y_series.to_numpy())

    metrics = evaluate_regression_model(model, X_df, y_series)

    assert "mse" in metrics
    assert "r2" in metrics
    assert isinstance(metrics["r2"], float)


def test_evaluate_regression_model_with_noise():
    """Test regression evaluation with noisy data."""
    X = np.random.rand(50, 3)
    y = X[:, 0] + X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 50)

    model = RandomForestRegressor(random_state=42, n_estimators=10)
    model.fit(X, y)

    metrics = evaluate_regression_model(model, X, y)

    # All metrics should be reasonable
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["r2"] <= 1  # R² can be negative for bad models


def test_plot_confusion_matrix():
    """Test confusion matrix plotting."""
    cm = np.array([[10, 2], [3, 15]])

    fig = plot_confusion_matrix(cm)

    assert isinstance(fig, plt.Figure)

    # Check that the plot was created (includes colorbar as second axis)
    axes = fig.get_axes()
    assert len(axes) == 2  # Main plot + colorbar

    # Clean up
    plt.close(fig)


def test_plot_confusion_matrix_with_class_names():
    """Test confusion matrix plotting with class names."""
    cm = np.array([[10, 2], [3, 15]])
    class_names = ["Class A", "Class B"]

    fig = plot_confusion_matrix(cm, class_names=class_names)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_confusion_matrix_save_to_file():
    """Test saving confusion matrix plot to file."""
    cm = np.array([[10, 2], [3, 15]])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "confusion_matrix.png"

        fig = plot_confusion_matrix(cm, output_path=output_path)

        # Check that file was created
        assert output_path.exists()
        plt.close(fig)


def test_plot_confusion_matrix_multiclass():
    """Test confusion matrix plotting for multiclass."""
    cm = np.array([[10, 2, 1], [3, 15, 2], [1, 1, 8]])
    class_names = ["A", "B", "C"]

    fig = plot_confusion_matrix(cm, class_names=class_names)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance():
    """Test feature importance plotting."""
    feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]
    importance_values = [0.4, 0.3, 0.2, 0.1]

    fig = plot_feature_importance(feature_names, importance_values)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance_top_n():
    """Test feature importance plotting with top N selection."""
    feature_names = [f"feature_{i}" for i in range(10)]
    importance_values = [0.1 * (10 - i) for i in range(10)]

    fig = plot_feature_importance(feature_names, importance_values, top_n=5)

    assert isinstance(fig, plt.Figure)

    # Check that only top 5 features are shown
    axes = fig.get_axes()
    ax = axes[0]
    # The number of bars should be 5
    assert len([p for p in ax.patches if p.get_height() > 0]) == 5

    plt.close(fig)


def test_plot_feature_importance_save_to_file():
    """Test saving feature importance plot to file."""
    feature_names = ["A", "B", "C"]
    importance_values = [0.5, 0.3, 0.2]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "feature_importance.png"

        fig = plot_feature_importance(
            feature_names, importance_values, output_path=output_path
        )

        # Check that file was created
        assert output_path.exists()
        plt.close(fig)


def test_save_evaluation_results():
    """Test saving evaluation results to JSON."""
    results = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1": 0.85,
        "confusion_matrix": [[10, 2], [3, 15]],
        "model_name": "LogisticRegression",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "evaluation_results.json"

        saved_path = save_evaluation_results(results, output_path)

        # Check that file was created
        assert saved_path.exists()
        assert saved_path == output_path

        # Load and verify contents
        import json

        with open(output_path) as f:
            loaded_results = json.load(f)

        assert loaded_results["accuracy"] == 0.85
        assert loaded_results["model_name"] == "LogisticRegression"


def test_save_evaluation_results_with_numpy():
    """Test saving evaluation results containing numpy arrays."""
    results = {
        "accuracy": np.float64(0.85),
        "scores": np.array([0.8, 0.9, 0.7]),
        "confusion_matrix": np.array([[10, 2], [3, 15]]),
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "results.json"

        saved_path = save_evaluation_results(results, output_path)

        assert saved_path.exists()

        # Load and verify that numpy types were converted
        import json

        with open(output_path) as f:
            loaded_results = json.load(f)

        assert isinstance(loaded_results["accuracy"], float)
        assert isinstance(loaded_results["scores"], list)
        assert isinstance(loaded_results["confusion_matrix"], list)


@patch("src.models.evaluation.logger")
def test_evaluation_logging(mock_logger):
    """Test that evaluation functions log appropriate messages."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # Test classification evaluation logging
    evaluate_classification_model(model, X, y)
    mock_logger.info.assert_any_call("Evaluating classification model")

    # Test regression evaluation logging
    reg_y = np.array([1.0, 2.0, 3.0])
    reg_model = LinearRegression()
    reg_model.fit(X, reg_y)

    evaluate_regression_model(reg_model, X, reg_y)
    mock_logger.info.assert_any_call("Evaluating regression model")


def test_integration_evaluation_pipeline():
    """Test integration of the complete evaluation pipeline."""
    # Create realistic classification dataset
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > X[:, 2] + X[:, 3]).astype(int)

    # Train model
    model = RandomForestClassifier(random_state=42, n_estimators=20)
    model.fit(X, y)

    # Evaluate classification
    class_metrics = evaluate_classification_model(model, X, y)

    # Get feature importance
    feature_names = [f"feature_{i}" for i in range(5)]
    importance_values = model.feature_importances_.tolist()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save evaluation results
        results_path = Path(temp_dir) / "classification_results.json"
        save_evaluation_results(class_metrics, results_path)

        # Plot and save confusion matrix
        cm_path = Path(temp_dir) / "confusion_matrix.png"
        cm_fig = plot_confusion_matrix(
            np.array(class_metrics["confusion_matrix"]), output_path=cm_path
        )

        # Plot and save feature importance
        fi_path = Path(temp_dir) / "feature_importance.png"
        fi_fig = plot_feature_importance(
            feature_names, importance_values, output_path=fi_path
        )

        # Verify all files were created
        assert results_path.exists()
        assert cm_path.exists()
        assert fi_path.exists()

        # Verify evaluation results
        assert class_metrics["accuracy"] > 0.5  # Should be better than random
        assert "confusion_matrix" in class_metrics
        assert len(class_metrics["confusion_matrix"]) == 2  # Binary classification

        # Clean up plots
        plt.close(cm_fig)
        plt.close(fi_fig)


def test_regression_evaluation_integration():
    """Test regression evaluation integration."""
    # Create regression dataset
    np.random.seed(42)
    X = np.random.rand(80, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 80)

    # Train model
    model = RandomForestRegressor(random_state=42, n_estimators=15)
    model.fit(X, y)

    # Evaluate
    reg_metrics = evaluate_regression_model(model, X, y)

    # Test feature importance plotting
    feature_names = ["feature_0", "feature_1", "feature_2"]
    importance_values = model.feature_importances_.tolist()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save results
        results_path = Path(temp_dir) / "regression_results.json"
        save_evaluation_results(reg_metrics, results_path)

        # Plot feature importance
        fi_fig = plot_feature_importance(feature_names, importance_values)

        # Verify results
        assert results_path.exists()
        assert reg_metrics["r2"] > 0.8  # Should have good fit
        assert reg_metrics["mse"] < 1.0  # Should have low error

        plt.close(fi_fig)
