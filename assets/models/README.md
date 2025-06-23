# Models Directory

This directory contains a local cache for trained models, model artifacts, and evaluation results.

## Structure

- **trained/**: Saved model files
- **evaluation/**: Model evaluation results
- **metadata/**: Model metadata and documentation
- **registry/**: Model registry information

## Purpose

The models directory serves to:

1. **Store models**: Maintain trained models in a consistent location
2. **Track versions**: Keep track of different model versions
3. **Document performance**: Store evaluation metrics and results
4. **Enable reproducibility**: Preserve model artifacts for reproducibility
5. **Facilitate deployment**: Support model deployment workflows

## Model Storage

Models can be stored in various formats:

- **Pickle/Joblib**: For scikit-learn models
- **SavedModel/HDF5**: For TensorFlow/Keras models
- **ONNX**: For interoperable models
- **PyTorch**: For PyTorch models
- **Custom formats**: For specialized models

## Model Metadata

Each model should include metadata such as:

- Training date and time
- Training dataset information
- Hyper-parameters
- Performance metrics
- Model architecture
- Dependencies and environment

Example metadata JSON:

```json
{
  "model_name": "random_forest_classifier",
  "version": "1.0.0",
  "description": "Random Forest model for customer churn prediction",
  "created_at": "2025-06-07T10:00:00Z",
  "author": "Your Name",
  "framework": "scikit-learn",
  "framework_version": "1.3.0",
  "python_version": "3.10.0",
  "training_dataset": "data/processed/customer_data_2025_06_01.parquet",
  "features": ["feature1", "feature2", "feature3", "..."],
  "target": "churn",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "random_state": 42
  },
  "metrics": {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.79,
    "f1_score": 0.80,
    "auc_roc": 0.88
  },
  "tags": ["classification", "churn", "customer"]
}
```

## Model Loading

Example code for loading a model:

```python
import joblib
from pathlib import Path

def load_model(model_name, version="latest"):
    """Load a trained model.

    Args:
        model_name: Name of the model
        version: Model version, defaults to "latest"

    Returns:
        Loaded model
    """
    if version == "latest":
        # Find the latest version
        model_dir = Path("models/trained")
        versions = [d for d in model_dir.glob(f"{model_name}_v*") if d.is_dir()]
        if not versions:
            raise FileNotFoundError(f"No versions found for model {model_name}")
        latest_version = sorted(versions)[-1]
        model_path = latest_version / "model.joblib"
    else:
        # Use the specified version
        model_path = Path(f"models/trained/{model_name}_v{version}/model.joblib")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)
```

## Model Evaluation

The evaluation directory contains results from model evaluation:

- Performance metrics
- Confusion matrices
- ROC curves
- Feature importance
- Evaluation reports

Example evaluation script:

```python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model(model, X_test, y_test, model_name, version):
    """Evaluate a model and save the results.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        version: Model version

    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Create evaluation directory
    eval_dir = Path(f"models/evaluation/{model_name}_v{version}")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "auc_roc": roc_auc
    }

    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(eval_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")

    # Save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(eval_dir / "roc_curve.png", dpi=300, bbox_inches="tight")

    return metrics
```

## Model Registry

For more advanced model management, consider using a model registry:

- **MLflow**: Open-source platform for managing the ML lifecycle
- **DVC**: Git for data and models
- **Weights & Biases**: ML experiment tracking and model registry
- **Neptune.ai**: Metadata store for MLOps

## Best Practices

1. **Version models**: Use semantic versioning for models
2. **Document everything**: Include comprehensive metadata
3. **Evaluate thoroughly**: Use multiple metrics for evaluation
4. **Track lineage**: Record the data and code used to create each model
5. **Monitor size**: Be mindful of model file sizes
6. **Backup models**: Implement a backup strategy for important models
7. **Automate evaluation**: Use automated workflows for model evaluation
8. **Standardize formats**: Use consistent formats for model storage
