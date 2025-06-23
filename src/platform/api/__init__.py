"""
FastAPI application for model serving and API endpoints.
"""

from .app import app, create_app
from .routes import router
from .schemas import HealthResponse, PredictionRequest, PredictionResponse

__all__ = [
    "create_app",
    "app",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
    "router",
]
