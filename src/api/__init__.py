"""
FastAPI application for model serving and API endpoints.
"""

from .app import app, create_app
from .models import HealthResponse, PredictionRequest, PredictionResponse
from .routes import router

__all__ = [
    "create_app",
    "app",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
    "router",
]
