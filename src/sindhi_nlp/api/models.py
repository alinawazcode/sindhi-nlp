"""Request and response models for the HTTP API (Pydantic).

The SDK uses plain dataclasses so it has no web dependencies; these models
describe the JSON that goes over the wire.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_TEXT_CHARS = 10_000
Text = Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEXT_CHARS)]

EXAMPLE_TEXT = "اڄ جو ڏينهن تمام سٺو آهي"


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"text": EXAMPLE_TEXT}]})

    text: Text


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"texts": [EXAMPLE_TEXT, "مون کي ڏاڍو ڏک ٿيو"]}]})

    texts: list[Text] = Field(min_length=1)


class PredictionItem(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    is_confident: bool


class PredictionResponse(PredictionItem):
    model: str


class BatchPredictionResponse(BaseModel):
    model: str
    count: int
    results: list[PredictionItem]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model: str | None = None
    backend: str | None = None
    package_version: str
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    name: str
    backend: str
    labels: list[str]
    confidence_type: str
    device: str
    max_length: int | None = None
    preprocessing_source: str
    seed: int | None = None
    base_model: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail