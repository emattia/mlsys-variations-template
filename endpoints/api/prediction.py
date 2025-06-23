"""Prediction API endpoint.

This module provides a FastAPI endpoint for making predictions with trained models.
"""

import logging
import os
from typing import Any

import polars as pl
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ml.inference import create_prediction_payload, predict
from src.ml.training import load_model
from src.platform.utils.common import get_model_path, setup_logging

# Load environment variables
load_dotenv()

# Configure logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", None),
    log_format=os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ),
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Model Prediction API",
    description="API for making predictions with trained machine learning models",
    version="1.0.0",
)


# Define request and response models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    data: dict[str, list[float]] = Field(
        ..., description="Dictionary mapping feature names to lists of feature values"
    )
    feature_names: list[str] | None = Field(
        None, description="List of feature names (optional if provided in data keys)"
    )
    model_name: str | None = Field(
        None,
        description="Model name (optional, defaults to first available model)",
    )
    return_probabilities: bool | None = Field(
        False, description="Whether to return probability estimates (for classifiers)"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    predictions: list[float] = Field(..., description="List of predictions")
    probabilities: list[list[float]] | None = Field(
        None,
        description="List of probability estimates for each class (for classifiers)",
    )
    model_info: dict[str, Any] = Field(
        ..., description="Information about the model used for prediction"
    )


# Global cache for loaded models
model_cache: dict[str, tuple[Any, Any]] = {}


def get_model(model_name: str | None = None) -> tuple[Any, Any]:
    """Get a model from the cache or load it from disk.

    Args:
        model_name: Name of the model to load (None for default)

    Returns:
        Tuple of (model, metadata)
    """
    global model_cache

    # If model_name is None, use the first available model
    if model_name is None:
        model_dir = get_model_path("trained")
        model_files = list(model_dir.glob("*.pkl"))

        if not model_files:
            raise HTTPException(status_code=404, detail="No models found")

        model_path = model_files[0]
        model_name = model_path.stem
    else:
        # Look for the model in the trained models directory
        model_dir = get_model_path("trained")
        model_path = model_dir / f"{model_name}.pkl"

        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

    # Check if model is in cache
    if model_name in model_cache:
        return model_cache[model_name]

    # Load model
    try:
        model, metadata = load_model(model_path)
        model_cache[model_name] = (model, metadata)
        logger.info(f"Loaded model '{model_name}' into cache")
        return model, metadata
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading model: {str(e)}"
        ) from e


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Model Prediction API"}


@app.get("/models")
def list_models() -> dict[str, list[dict[str, Any]]]:
    """List available models."""
    model_dir = get_model_path("trained")
    model_files = list(model_dir.glob("*.pkl"))

    models = []
    for model_path in model_files:
        try:
            _, metadata = load_model(model_path)
            models.append(
                {
                    "name": model_path.stem,
                    "path": str(model_path),
                    "type": metadata.get("model_type", "unknown"),
                    "problem_type": metadata.get("problem_type", "unknown"),
                    "feature_columns": metadata.get("feature_columns", []),
                    "target_column": metadata.get("target_column", None),
                }
            )
        except Exception as e:
            logger.error(f"Error loading model '{model_path}': {e}")
            models.append(
                {"name": model_path.stem, "path": str(model_path), "error": str(e)}
            )

    return {"models": models}


@app.get("/models/{model_name}")
def get_model_info(model_name: str) -> dict[str, Any]:
    """Get information about a specific model."""
    try:
        _, metadata = get_model(model_name)
        return {"name": model_name, "metadata": metadata}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting model info: {str(e)}"
        ) from e


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest) -> dict[str, Any]:
    """Make predictions with a trained model."""
    try:
        # Get model
        model, metadata = get_model(request.model_name)

        # Create feature matrix
        try:
            payload = create_prediction_payload(request.data, request.feature_names)
            feature_names = payload["feature_names"]

            # Convert to DataFrame
            data_dict = payload["data"]
            df = pl.DataFrame({col: data_dict[col] for col in feature_names})
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error creating feature matrix: {str(e)}"
            ) from e

        # Make predictions
        if request.return_probabilities:
            try:
                y_pred, y_prob = predict(model, df, return_probabilities=True)
                has_probabilities = True
            except Exception:
                y_pred = predict(model, df, return_probabilities=False)
                y_prob = None
                has_probabilities = False
        else:
            y_pred = predict(model, df, return_probabilities=False)
            y_prob = None
            has_probabilities = False

        # Convert predictions to list
        predictions = y_pred.tolist()

        # Convert probabilities to list if available
        probabilities = None
        if has_probabilities and y_prob is not None:
            probabilities = y_prob.tolist()

        # Create response
        response = {
            "predictions": predictions,
            "probabilities": probabilities,
            "model_info": {
                "name": request.model_name or "default",
                "type": metadata.get("model_type", "unknown"),
                "problem_type": metadata.get("problem_type", "unknown"),
                "feature_columns": feature_names,
            },
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error making predictions: {str(e)}"
        ) from e


if __name__ == "__main__":
    import uvicorn

    # Run the API server
    uvicorn.run(
        "endpoints.api.prediction:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "False").lower() == "true",
    )
