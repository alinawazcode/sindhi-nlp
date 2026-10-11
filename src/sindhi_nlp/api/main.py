"""FastAPI application entry point.

Run from the project root:

    PYTHONPATH=src uvicorn sindhi_nlp.api.main:app --host 127.0.0.1 --port 8000

The model named in configs/inference.yaml is loaded once at startup. Interactive
documentation is served at /docs.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..inference.predictor import SentimentPredictor
from ..inference.schemas import InvalidInputError
from .dependencies import BatchTooLargeError, ModelService, ModelUnavailableError, Settings, get_settings
from .routes import router

logger = logging.getLogger("sindhi_nlp.api")

HTTP_CODES = {404: "not_found", 405: "method_not_allowed", 413: "payload_too_large"}


def error_response(status: int, code: str, message: str, details: list[dict[str, str]] | None = None) -> JSONResponse:
    """The single error format used by every endpoint."""
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field paths and messages only: the submitted text is never echoed back or logged.
        details = [
            {"field": ".".join(str(part) for part in err["loc"][1:]) or "body", "message": err["msg"]}
            for err in exc.errors()
        ]
        return error_response(422, "validation_error", "The request is not valid", details)

    @app.exception_handler(InvalidInputError)
    async def _invalid(request: Request, exc: InvalidInputError) -> JSONResponse:
        return error_response(422, "invalid_input", str(exc))

    @app.exception_handler(BatchTooLargeError)
    async def _too_large(request: Request, exc: BatchTooLargeError) -> JSONResponse:
        return error_response(422, "batch_too_large", str(exc))

    @app.exception_handler(ModelUnavailableError)
    async def _unavailable(request: Request, exc: ModelUnavailableError) -> JSONResponse:
        return error_response(503, "model_unavailable", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = HTTP_CODES.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", "An unexpected error occurred")


def create_app(predictor: SentimentPredictor | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Normally the model is loaded from the inference config when the server starts.
    Pass ``predictor`` to use an already-loaded one (handy in tests).
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.service is None:
            app.state.service = ModelService.from_settings(settings)
        yield

    app = FastAPI(
        title="Sindhi sentiment API",
        version=__version__,
        description="Classify Sindhi text as negative, neutral, or positive.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = ModelService(predictor) if predictor is not None else None

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        logger.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code,
                    1000 * (time.perf_counter() - start))
        return response

    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()