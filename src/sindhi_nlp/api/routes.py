"""HTTP endpoints: health, model info, and prediction."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .. import __version__
from .dependencies import BatchTooLargeError, ModelService, Settings
from .models import (
    BatchPredictionResponse,
    BatchPredictRequest,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionItem,
    PredictionResponse,
    PredictRequest,
)

router = APIRouter()

UNAVAILABLE: dict[int | str, dict[str, Any]] = {503: {"model": ErrorResponse, "description": "The model is not loaded"}}
INVALID: dict[int | str, dict[str, Any]] = {422: {"model": ErrorResponse, "description": "The input is invalid"}}


def get_service(request: Request) -> ModelService:
    """FastAPI dependency returning the model service stored on the application."""
    service = getattr(request.app.state, "service", None)
    return service if service is not None else ModelService(error="The service is still starting")


def get_settings(request: Request) -> Settings:
    """FastAPI dependency returning the settings stored on the application."""
    return request.app.state.settings


def _item(prediction) -> PredictionItem:
    return PredictionItem(**prediction.to_dict())


@router.get("/health", response_model=HealthResponse, responses=UNAVAILABLE, tags=["status"])
def health(service: ModelService = Depends(get_service)):
    """Report whether the model is loaded and which one it is."""
    if not service.loaded:
        body = HealthResponse(
            status="unavailable", model_loaded=False, package_version=__version__, detail=service.error
        )
        return JSONResponse(status_code=503, content=body.model_dump())
    info = service.info
    return HealthResponse(
        status="ok", model_loaded=True, model=info.name, backend=info.backend, package_version=__version__
    )


@router.get("/model", response_model=ModelInfoResponse, responses=UNAVAILABLE, tags=["status"])
def model_info(service: ModelService = Depends(get_service)):
    """Details of the loaded model: labels, backend, token limit, preprocessing."""
    return ModelInfoResponse(**service.info.to_dict())


@router.post("/predict", response_model=PredictionResponse, responses={**INVALID, **UNAVAILABLE}, tags=["prediction"])
def predict(body: PredictRequest, service: ModelService = Depends(get_service)):
    """Classify one Sindhi text as negative, neutral, or positive."""
    prediction = service.predict(body.text)
    return PredictionResponse(**_item(prediction).model_dump(), model=service.info.name)


@router.post(
    "/predict/batch", response_model=BatchPredictionResponse, responses={**INVALID, **UNAVAILABLE}, tags=["prediction"]
)
def predict_batch(
    body: BatchPredictRequest,
    service: ModelService = Depends(get_service),
    settings: Settings = Depends(get_settings),
):
    """Classify several texts. Results keep the order of the request."""
    if len(body.texts) > settings.max_batch_size:
        raise BatchTooLargeError(len(body.texts), settings.max_batch_size)
    predictions = service.predict_batch(body.texts)
    return BatchPredictionResponse(
        model=service.info.name, count=len(predictions), results=[_item(p) for p in predictions]
    )
