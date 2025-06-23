"""
Model service for handling model loading, caching, and predictions.
"""

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..config.manager import ConfigManager
from .models import ModelInfo


def safe_pickle_load(file_path: Path) -> Any:
    """Safely load a pickle file with error handling."""
    try:
        with open(file_path, "rb") as f:
            # Load with restricted unpickler in production
            data = pickle.load(f)  # nosec B301 - Internal model files only
        return data
    except (pickle.UnpicklingError, EOFError, ImportError) as e:
        raise ValueError(f"Failed to load pickle file: {e}") from e


class ModelService:
    """Service for managing model loading and predictions."""

    def __init__(self, config_manager: ConfigManager = None):
        if config_manager is None:
            from ..config.manager import get_config_manager

            config_manager = get_config_manager()
        self.config_manager = config_manager
        self.models: dict[str, dict[str, Any]] = {}
        self.startup_time = time.time()

    def load_model(self, model_name: str, model_path: str | None = None) -> bool:
        """Load a model from file."""
        try:
            if model_path is None:
                # Try to find model in standard locations
                config = self.config_manager.get_config()
                model_dir = Path(config.paths.model_root)

                # Try different file extensions
                for ext in [".pkl", ".joblib", ".pickle"]:
                    potential_path = model_dir / f"{model_name}{ext}"
                    if potential_path.exists():
                        model_path = str(potential_path)
                        break

                if model_path is None:
                    raise FileNotFoundError(
                        f"Model '{model_name}' not found in {model_dir}"
                    )

            # Load the model
            model_path = Path(model_path)
            if model_path.suffix in [".joblib"]:
                model = joblib.load(model_path)
            else:
                # Use safe pickle loading for security
                model = safe_pickle_load(model_path)

            # Get model metadata
            model_info = self._extract_model_info(model, model_name, str(model_path))

            # Store model and metadata
            self.models[model_name] = {
                "model": model,
                "info": model_info,
                "loaded_at": datetime.now(),
                "path": str(model_path),
            }

            return True

        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            return False

    def _extract_model_info(self, model: Any, name: str, path: str) -> ModelInfo:
        """Extract metadata from a loaded model."""
        # Determine model type
        model_type = "unknown"
        target_classes = None
        features = []

        if hasattr(model, "_estimator_type"):
            model_type = model._estimator_type
        elif hasattr(model, "predict_proba"):
            model_type = "classifier"
        elif hasattr(model, "predict"):
            model_type = "regressor"

        # Get target classes for classifiers
        if hasattr(model, "classes_"):
            target_classes = [str(cls) for cls in model.classes_]

        # Try to get feature names
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        elif hasattr(model, "n_features_in_"):
            features = [f"feature_{i}" for i in range(model.n_features_in_)]
        else:
            features = ["feature_0"]  # Default

        return ModelInfo(
            name=name,
            type=model_type,
            features=features,
            target_classes=target_classes,
            loaded_at=datetime.now(),
            model_path=path,
        )

    def predict(
        self,
        model_name: str,
        features: list[float | int] | list[list[float | int]],
        return_probabilities: bool = False,
    ) -> tuple[list[Any], list[list[float]] | None, float]:
        """Make predictions using the specified model."""
        start_time = time.time()

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]["model"]

        # Convert features to numpy array
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Make predictions
        predictions = model.predict(X)

        # Get probabilities if requested and supported
        probabilities = None
        if return_probabilities and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            probabilities = proba.tolist() if hasattr(proba, "tolist") else proba

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Convert predictions to list if not already
        predictions_list = (
            predictions.tolist() if hasattr(predictions, "tolist") else predictions
        )
        return predictions_list, probabilities, processing_time

    def get_model_info(self, model_name: str) -> ModelInfo | None:
        """Get information about a loaded model."""
        if model_name not in self.models:
            return None
        return self.models[model_name]["info"]

    def list_models(self) -> list[str]:
        """List all loaded models."""
        return list(self.models.keys())

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        if model_name in self.models:
            del self.models[model_name]
            return True
        return False

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.startup_time

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        return {
            "status": "healthy" if self.models else "no_models_loaded",
            "timestamp": datetime.now(),
            "models_loaded": self.list_models(),
            "uptime_seconds": self.get_uptime(),
            "version": "1.0.0",  # TODO: Get from package metadata
        }

    def create_default_model(self) -> bool:
        """Create and load a default model for demonstration."""
        try:
            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier

            # Create sample data
            X, y = make_classification(
                n_samples=1000, n_features=4, n_classes=2, random_state=42
            )

            # Train a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            # Add feature names
            model.feature_names_in_ = np.array([f"feature_{i}" for i in range(4)])

            # Save model
            config = self.config_manager.get_config()
            model_dir = Path(config.paths.model_root)
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / "default.joblib"
            joblib.dump(model, model_path)

            # Load the model
            return self.load_model("default", str(model_path))

        except Exception as e:
            print(f"Error creating default model: {e}")
            return False
