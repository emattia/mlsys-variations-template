"""
Pydantic models for API request and response validation.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "features": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]],
                "model_name": "default",
                "return_probabilities": False,
            }
        },
    )

    features: list[list[float | int]] = Field(
        ..., description="Input features for prediction", min_length=1
    )
    model_name: str = Field(
        default="default", description="Name of the model to use for prediction"
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return prediction probabilities (for classification)",
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predictions": [0, 1],
                "probabilities": [[0.9, 0.1], [0.3, 0.7]],
                "model_name": "default",
                "status": "success",
            }
        },
    )

    predictions: list[float | int | str] = Field(..., description="Model predictions")
    probabilities: list[list[float]] | None = Field(
        default=None, description="Prediction probabilities (for classification models)"
    )
    model_name: str = Field(..., description="Name of the model used")
    status: str = Field(..., description="Status of the prediction")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "features": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]],
                "model_name": "default",
                "return_probabilities": False,
            }
        },
    )

    features: list[list[float | int]] = Field(
        ..., description="Batch of input features for prediction", min_length=1
    )
    model_name: str = Field(
        default="default", description="Name of the model to use for prediction"
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return prediction probabilities (for classification)",
    )


class HealthResponse(BaseModel):
    """Response model for health checks."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0",
                "models_loaded": ["default"],
                "uptime_seconds": 3600.5,
            }
        },
    )

    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    models_loaded: list[str] = Field(..., description="List of loaded models")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information response."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "name": "default",
                "type": "classification",
                "features": ["feature1", "feature2", "feature3", "feature4"],
                "target_classes": ["class1", "class2"],
                "loaded_at": "2024-01-01T12:00:00Z",
                "model_path": "/models/default.pkl",
            }
        },
    )

    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (classification, regression, etc.)")
    features: list[str] = Field(..., description="Expected input features")
    target_classes: list[str] | None = Field(
        default=None, description="Target classes (for classification models)"
    )
    loaded_at: datetime = Field(..., description="Model load timestamp")
    model_path: str = Field(..., description="Path to model file")


class ErrorResponse(BaseModel):
    """Error response model."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "error": "Invalid input",
                "detail": "Feature vector length mismatch",
                "timestamp": "2024-01-01T12:00:00Z",
            }
        },
    )

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")
