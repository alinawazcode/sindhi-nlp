"""Settings, and construction and caching of the predictor used by the routes."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from ..inference.predictor import SentimentPredictor
from ..inference.schemas import ModelInfo, Prediction
from ..utils.config import ConfigError, load_yaml
from ..utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "configs/inference.yaml"
DEFAULT_MAX_BATCH_SIZE = 64


class ModelUnavailableError(RuntimeError):
    """The model is not loaded (still starting, or failed to load)."""


class BatchTooLargeError(ValueError):
    """A batch request has more texts than the configured limit."""

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"Too many texts: {size} (the limit is {limit} per request)")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the HTTP service, read from the inference config and environment."""

    config_path: Path
    project_root: Path
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    cors_origins: tuple[str, ...] = ()


def load_settings(config_path: str | Path | None = None, project_root: str | Path | None = None) -> Settings:
    """Read API settings from ``configs/inference.yaml`` (the ``api:`` section).

    Environment variables SINDHI_NLP_CONFIG and SINDHI_NLP_ROOT set the defaults.
    A missing config file is not an error here; the model will simply fail to load
    and the API will report itself as unavailable.
    """
    root = Path(project_root or os.environ.get("SINDHI_NLP_ROOT", ".")).resolve()
    path = Path(config_path or os.environ.get("SINDHI_NLP_CONFIG", DEFAULT_CONFIG))
    if not path.is_absolute():
        path = root / path

    api: dict = {}
    if path.is_file():
        api = load_yaml(path).get("api") or {}
        if not isinstance(api, dict):
            raise ConfigError("'api' in the inference config must be a mapping")
    else:
        logger.warning("Inference config not found: %s", path)

    max_batch = api.get("max_batch_size", DEFAULT_MAX_BATCH_SIZE)
    if isinstance(max_batch, bool) or not isinstance(max_batch, int) or max_batch < 1:
        raise ConfigError("api.max_batch_size must be a positive integer")
    origins = api.get("cors_origins") or []
    if not isinstance(origins, list) or not all(isinstance(o, str) for o in origins):
        raise ConfigError("api.cors_origins must be a list of strings")
    return Settings(path, root, max_batch, tuple(origins))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings for the module-level app, read once."""
    return load_settings()


class ModelService:
    """Holds the loaded predictor (or the reason it is missing) and serializes inference."""

    def __init__(self, predictor: SentimentPredictor | None = None, error: str | None = None) -> None:
        self._predictor = predictor
        self.error = error
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelService":
        """Load the model once. A failure is logged and kept, so the API can still answer /health."""
        try:
            predictor = SentimentPredictor.from_config(settings.config_path, settings.project_root)
        except Exception:
            logger.exception("The model could not be loaded")
            return cls(error="The model failed to load. See the server log for details.")
        logger.info("Loaded model %s (%s)", predictor.info.name, predictor.info.backend)
        return cls(predictor)

    @property
    def loaded(self) -> bool:
        """True when a predictor is loaded and ready to serve requests."""
        return self._predictor is not None

    @property
    def info(self) -> ModelInfo:
        """Details of the loaded model. Raises ``ModelUnavailableError`` if none is loaded."""
        return self.require().info

    def require(self) -> SentimentPredictor:
        """Return the loaded predictor, or raise ``ModelUnavailableError`` explaining why there is none."""
        if self._predictor is None:
            raise ModelUnavailableError(self.error or "The model is not loaded yet")
        return self._predictor

    def predict(self, text: str) -> Prediction:
        """Classify one text under the model lock, so concurrent requests never share the model unsafely."""
        predictor = self.require()
        with self._lock:
            return predictor.predict(text)

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify several texts under the model lock, keeping the input order."""
        predictor = self.require()
        with self._lock:  # one inference at a time; model objects are not guaranteed thread-safe
            return predictor.predict_batch(texts, errors="raise")  # type: ignore[return-value]
