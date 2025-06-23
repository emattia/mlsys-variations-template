"""
FastAPI application factory and configuration.
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from ..config.manager import get_config_manager
from .models import ErrorResponse
from .routes import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting FastAPI application...")

    if not hasattr(app.state, "config"):
        logger.info("No config found in app.state, initializing...")
        try:
            config_manager = get_config_manager()
            config = config_manager.load_config()
            app.state.config = config
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Creating default configuration as fallback...")
            config_manager.create_default_config_files()
            app.state.config = config_manager.load_config()

    if not hasattr(app.state, "model_service"):
        logger.info("No model_service found in app.state, initializing...")
        from .service import ModelService  # Local import to avoid circular

        config_manager = get_config_manager()
        if not config_manager.get_config():
            config_manager.load_config()
        app.state.model_service = ModelService(config_manager)

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application...")


def create_app(config_override: dict[str, Any] = None) -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="MLOps Template API",
        description="REST API for machine learning model serving and management",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # Configure appropriately for production
    )

    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

    # Add exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        error_response = ErrorResponse(
            error=exc.detail,
            detail=getattr(exc, "detail", None),
            timestamp=datetime.now(),
        )
        return JSONResponse(
            status_code=exc.status_code, content=error_response.model_dump(mode="json")
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        error_response = ErrorResponse(
            error="Internal server error",
            detail=str(exc) if app.debug else "An unexpected error occurred",
            timestamp=datetime.now(),
        )
        return JSONResponse(
            status_code=500, content=error_response.model_dump(mode="json")
        )

    # Include routers
    app.include_router(router, prefix="/api/v1")

    # Add direct endpoints for backward compatibility (tests expect these)
    from .service import ModelService
    from .models import PredictionRequest, PredictionResponse, HealthResponse

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check_direct():
        """Health check endpoint (direct access)."""
        try:
            config_manager = get_config_manager()
            service = ModelService(config_manager)
            health_data = service.health_check()
            return HealthResponse(**health_data)
        except Exception as e:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Health check failed: {str(e)}",
            )

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    async def predict_direct(request: PredictionRequest):
        """Make a prediction (direct access)."""
        try:
            config_manager = get_config_manager()
            service = ModelService(config_manager)

            # Ensure model is loaded
            if request.model_name not in service.list_models():
                # Try to load default model if it doesn't exist
                if request.model_name == "default":
                    success = service.create_default_model()
                    if not success:
                        from fastapi import HTTPException, status

                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Model '{request.model_name}' not found and failed to create default",
                        )
                else:
                    from fastapi import HTTPException, status

                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model '{request.model_name}' not found",
                    )

            predictions, probabilities, processing_time = service.predict(
                request.model_name, request.features, request.return_probabilities
            )

            return PredictionResponse(
                predictions=predictions,
                probabilities=probabilities,
                model_name=request.model_name,
                status="success",
            )

        except Exception as e:
            from fastapi import HTTPException, status

            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
                )
            elif "validation" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {str(e)}",
                )

    @app.get("/models", response_model=list[str], tags=["models"])
    async def list_models_direct():
        """List all loaded models (direct access)."""
        try:
            config_manager = get_config_manager()
            service = ModelService(config_manager)
            return service.list_models()
        except Exception as e:
            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list models: {str(e)}",
            )

    # Add root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "MLOps Template API",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/api/v1/health",
        }

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="MLOps Template API",
            version="1.0.0",
            description="""
            A comprehensive REST API for machine learning model serving and management.

            ## Features
            - Model loading and management
            - Single and batch predictions
            - Health monitoring
            - Model metadata and information
            - Automatic model creation for demonstrations

            ## Usage
            1. Check service health: `GET /api/v1/health`
            2. Create default model: `POST /api/v1/models/default/create`
            3. Make predictions: `POST /api/v1/predict`
            4. List models: `GET /api/v1/models`
            """,
            routes=app.routes,
        )

        # Add custom tags
        openapi_schema["tags"] = [
            {"name": "root", "description": "Root endpoints"},
            {"name": "health", "description": "Health check endpoints"},
            {"name": "models", "description": "Model management endpoints"},
            {"name": "prediction", "description": "Prediction endpoints"},
        ]

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app


# Create the application instance for use by tests and other modules
app = create_app()


# Module-level model_service that lazily initializes
class _ModelServiceProxy:
    """Proxy to lazily initialize model_service."""

    def __init__(self):
        self._instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = self._create_model_service()
        return getattr(self._instance, name)

    def _create_model_service(self):
        """Create the model service instance."""
        from .service import ModelService
        from ..config.manager import get_config_manager

        config_manager = get_config_manager()
        try:
            if not config_manager.get_config():
                config_manager.load_config()
        except RuntimeError:
            # If no config loaded, load default config
            config_manager.load_config()

        return ModelService(config_manager)


# Create proxy instance that will be accessible as model_service
model_service = _ModelServiceProxy()


def get_model_service():
    """Get the model service instance."""
    return model_service._instance or model_service._create_model_service()


if __name__ == "__main__":
    import uvicorn

    # Run the application using the factory
    uvicorn.run(
        "src.api.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
