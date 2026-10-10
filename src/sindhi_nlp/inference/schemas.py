"""Typed result objects and errors for the prediction interface."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class InvalidInputError(ValueError):
    """The text cannot be classified (wrong type, empty, or too long)."""


class ModelNotFoundError(FileNotFoundError):
    """The model folder or its model files do not exist."""


class ModelLoadError(RuntimeError):
    """The model files exist but could not be loaded."""


@dataclass(frozen=True)
class Prediction:
    """One classified text.

    ``probabilities`` sum to 1 and are keyed by label. For the Hugging Face
    model they are softmax probabilities. For the TF-IDF + LinearSVC baseline
    they are a softmax of the classifier margins: a useful ranking score, but
    not a calibrated probability (see ``ModelInfo.confidence_type``).
    """

    text: str
    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    is_confident: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    backend: str                      # "sklearn" or "transformers"
    model_dir: str
    labels: tuple[str, ...]
    confidence_type: str              # "probability" or "score"
    device: str = "cpu"
    max_length: int | None = None     # token limit (transformers only)
    preprocessing_source: str = "defaults"
    seed: int | None = None
    base_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = list(self.labels)
        return data