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
    """Body of ``POST /predict``: one ``text``."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"text": EXAMPLE_TEXT}]})

    text: Text


class BatchPredictRequest(BaseModel):
    """Body of ``POST /predict/batch``: a list of ``texts``."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"texts": [EXAMPLE_TEXT, "مون کي ڏاڍو ڏک ٿيو"]}]})

    texts: list[Text] = Field(min_length=1)


class PredictionItem(BaseModel):
    """One classified text: label, confidence and per-class probabilities."""

    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    is_confident: bool


class PredictionResponse(PredictionItem):
    """Response of ``POST /predict``: a prediction plus the model that produced it."""

    model: str


class BatchPredictionResponse(BaseModel):
    """Response of ``POST /predict/batch``: predictions in input order plus the model name."""

    model: str
    count: int
    results: list[PredictionItem]


class HealthResponse(BaseModel):
    """Response of ``GET /health``: service status and the loaded model."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model: str | None = None
    backend: str | None = None
    package_version: str
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    """Response of ``GET /model``: details of the loaded model."""

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
    """One field-level problem inside an error response."""

    code: str
    message: str
    details: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    """The JSON shape of every error: a short ``code``, a ``message`` and optional ``details``."""

    error: ErrorDetail
