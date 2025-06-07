"""
FastAPI routes for model serving endpoints.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from ..config.manager import ConfigManager
from .models import (
    BatchPredictionRequest,
    HealthResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from .service import ModelService

# Create router
router = APIRouter()


# Dependency to get model service
def get_model_service() -> ModelService:
    """Dependency to get model service instance."""
    config_manager = ConfigManager()
    return ModelService(config_manager)


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(service: ModelService = Depends(get_model_service)):
    """Health check endpoint."""
    try:
        health_data = service.health_check()
        return HealthResponse(**health_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )


@router.get("/models", response_model=list[str], tags=["models"])
async def list_models(service: ModelService = Depends(get_model_service)):
    """List all loaded models."""
    try:
        return service.list_models()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@router.get("/models/{model_name}", response_model=ModelInfo, tags=["models"])
async def get_model_info(
    model_name: str, service: ModelService = Depends(get_model_service)
):
    """Get information about a specific model."""
    try:
        model_info = service.get_model_info(model_name)
        if model_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found",
            )
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@router.post("/models/{model_name}/load", tags=["models"])
async def load_model(
    model_name: str,
    model_path: str = None,
    service: ModelService = Depends(get_model_service),
):
    """Load a model."""
    try:
        success = service.load_model(model_name, model_path)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load model '{model_name}'",
            )
        return {"message": f"Model '{model_name}' loaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}",
        )


@router.delete("/models/{model_name}", tags=["models"])
async def unload_model(
    model_name: str, service: ModelService = Depends(get_model_service)
):
    """Unload a model."""
    try:
        success = service.unload_model(model_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found",
            )
        return {"message": f"Model '{model_name}' unloaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error unloading model: {str(e)}",
        )


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    request: PredictionRequest, service: ModelService = Depends(get_model_service)
):
    """Make a single prediction."""
    try:
        # Ensure model is loaded
        if request.model_name not in service.list_models():
            # Try to load default model if it doesn't exist
            if request.model_name == "default":
                success = service.create_default_model()
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model '{request.model_name}' not found and failed to create default",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model_name}' not found",
                )

        predictions, probabilities, processing_time = service.predict(
            request.features, request.model_name, request.return_probabilities
        )

        return PredictionResponse(
            prediction=predictions,
            probabilities=probabilities,
            model_name=request.model_name,
            timestamp=datetime.now(),
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post("/predict/batch", response_model=PredictionResponse, tags=["prediction"])
async def predict_batch(
    request: BatchPredictionRequest, service: ModelService = Depends(get_model_service)
):
    """Make batch predictions."""
    try:
        # Ensure model is loaded
        if request.model_name not in service.list_models():
            # Try to load default model if it doesn't exist
            if request.model_name == "default":
                success = service.create_default_model()
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model '{request.model_name}' not found and failed to create default",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model_name}' not found",
                )

        predictions, probabilities, processing_time = service.predict(
            request.features, request.model_name, request.return_probabilities
        )

        return PredictionResponse(
            prediction=predictions,
            probabilities=probabilities,
            model_name=request.model_name,
            timestamp=datetime.now(),
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@router.post("/models/default/create", tags=["models"])
async def create_default_model(service: ModelService = Depends(get_model_service)):
    """Create and load a default demonstration model."""
    try:
        success = service.create_default_model()
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create default model",
            )
        return {"message": "Default model created and loaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating default model: {str(e)}",
        )
