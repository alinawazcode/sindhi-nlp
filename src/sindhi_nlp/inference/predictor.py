"""A stable interface for classifying Sindhi text with a trained model.

    predictor = SentimentPredictor.from_pretrained("artifacts/models/xlmr_mixed")
    result = predictor.predict("اڄ جو ڏينهن تمام سٺو آهي")
    result.label, result.confidence, result.probabilities

Two model types are supported, detected from the files in the model folder:
the scikit-learn baseline (``model.joblib``) and a fine-tuned Hugging Face
model (``model.safetensors``). torch and transformers are imported only when
a Hugging Face model is loaded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np

from ..utils.config import ConfigError, load_config
from .preprocessing import TextPreprocessor
from .schemas import InvalidInputError, ModelInfo, ModelLoadError, ModelNotFoundError, Prediction

SKLEARN_FILE = "model.joblib"
TRANSFORMERS_FILES = ("model.safetensors", "pytorch_model.bin")
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32


@dataclass
class Backend:
    """What the predictor needs from a model: label names and a scoring function."""

    kind: str                                         # "sklearn" | "transformers"
    labels: list[str]
    score_fn: Callable[[list[str]], np.ndarray]       # texts -> (n, num_labels) raw scores
    confidence_type: str                              # "probability" | "score"
    device: str = "cpu"
    max_length: int | None = None


def softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, stable for large values."""
    scores = np.asarray(scores, dtype=float)
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_sklearn_backend(model_dir: Path) -> Backend:
    import joblib

    try:
        pipeline = joblib.load(model_dir / SKLEARN_FILE)
        labels = [str(label) for label in pipeline.classes_]
    except Exception as exc:  # corrupt file, version mismatch, wrong object type
        raise ModelLoadError(f"Could not load {model_dir / SKLEARN_FILE}: {exc}") from exc

    if hasattr(pipeline, "decision_function"):
        def score_fn(texts: list[str]) -> np.ndarray:
            scores = np.asarray(pipeline.decision_function(texts), dtype=float)
            return np.column_stack([-scores, scores]) if scores.ndim == 1 else scores

        return Backend("sklearn", labels, score_fn, confidence_type="score")
    if hasattr(pipeline, "predict_proba"):
        def proba_fn(texts: list[str]) -> np.ndarray:
            return np.log(np.clip(pipeline.predict_proba(texts), 1e-12, 1.0))

        return Backend("sklearn", labels, proba_fn, confidence_type="probability")
    raise ModelLoadError("The saved model has neither decision_function nor predict_proba")


def _labels_from_mapping(model_dir: Path) -> list[str] | None:
    mapping = _read_json(model_dir / "label_mapping.json")
    if not mapping:
        return None
    try:
        return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    except (TypeError, ValueError):
        return None


def _load_transformers_backend(model_dir: Path, device: str, max_length: int) -> Backend:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ModelLoadError("Hugging Face models need: pip install torch transformers") from exc

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("device='cuda' was requested but no CUDA GPU is available")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval().to(resolved)
    except Exception as exc:
        raise ModelLoadError(f"Could not load the model in {model_dir}: {exc}") from exc

    id2label = {int(i): str(name) for i, name in model.config.id2label.items()}
    labels = _labels_from_mapping(model_dir) or [id2label[i] for i in sorted(id2label)]
    if len(labels) != model.config.num_labels:
        raise ModelLoadError(
            f"label_mapping.json lists {len(labels)} labels but the model has {model.config.num_labels} outputs"
        )

    def score_fn(texts: list[str]) -> np.ndarray:
        batch = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(resolved)
        with torch.no_grad():
            logits = model(**batch).logits
        return logits.float().cpu().numpy()

    return Backend("transformers", labels, score_fn, confidence_type="probability", device=resolved, max_length=max_length)


class SentimentPredictor:
    """Classify Sindhi sentences as negative, neutral, or positive."""

    def __init__(
        self,
        backend: Backend,
        preprocessor: TextPreprocessor,
        info: ModelInfo,
        batch_size: int = DEFAULT_BATCH_SIZE,
        confidence_threshold: float | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self._backend = backend
        self._preprocessor = preprocessor
        self._info = info
        self._batch_size = batch_size
        self._threshold = confidence_threshold

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int | None = None,
        confidence_threshold: float | None = None,
        max_chars: int | None = None,
    ) -> "SentimentPredictor":
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise ModelNotFoundError(f"Model folder not found: {model_dir}")

        training = _read_json(model_dir / "training_config.json")
        run_config = _read_json(model_dir / "config.json")
        saved_length = ((training.get("config") or {}).get("tokenizer") or {}).get("max_length")

        if (model_dir / SKLEARN_FILE).is_file():
            backend = _load_sklearn_backend(model_dir)
        elif any((model_dir / name).is_file() for name in TRANSFORMERS_FILES):
            backend = _load_transformers_backend(model_dir, device, int(max_length or saved_length or DEFAULT_MAX_LENGTH))
        else:
            raise ModelNotFoundError(
                f"No model files in {model_dir}: expected {SKLEARN_FILE} or one of {', '.join(TRANSFORMERS_FILES)}"
            )

        preprocessor = (
            TextPreprocessor.from_model_dir(model_dir, max_chars) if max_chars else TextPreprocessor.from_model_dir(model_dir)
        )
        info = ModelInfo(
            name=model_dir.name,
            backend=backend.kind,
            model_dir=str(model_dir),
            labels=tuple(backend.labels),
            confidence_type=backend.confidence_type,
            device=backend.device,
            max_length=backend.max_length,
            preprocessing_source=preprocessor.source,
            seed=training.get("seed", run_config.get("seed")),
            base_model=(training.get("config") or {}).get("model_name"),
        )
        return cls(backend, preprocessor, info, batch_size, confidence_threshold)

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/inference.yaml", project_root: str | Path = ".") -> "SentimentPredictor":
        """Build a predictor from ``configs/inference.yaml``."""
        root = Path(project_root).resolve()
        cfg = load_config(config_path, required=("model_dir",))
        if cfg.get("device", "auto") not in ("auto", "cpu", "cuda"):
            raise ConfigError("device must be 'auto', 'cpu', or 'cuda'")
        return cls.from_pretrained(
            root / cfg["model_dir"],
            device=cfg.get("device", "auto"),
            batch_size=int(cfg.get("batch_size") or DEFAULT_BATCH_SIZE),
            max_length=cfg.get("max_length"),
            confidence_threshold=cfg.get("confidence_threshold"),
            max_chars=cfg.get("max_chars"),
        )

    # --------------------------------------------------------------- properties

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def labels(self) -> tuple[str, ...]:
        return self._info.labels

    # --------------------------------------------------------------- prediction

    def predict(self, text: str) -> Prediction:
        """Classify one text. Raises ``InvalidInputError`` for empty or unsuitable input."""
        result = self.predict_batch([text])[0]
        assert result is not None  # errors="raise" never returns None
        return result

    def predict_batch(
        self,
        texts: Iterable[str],
        batch_size: int | None = None,
        errors: Literal["raise", "none"] = "raise",
    ) -> list[Prediction | None]:
        """Classify many texts, keeping the input order.

        With ``errors="raise"`` the first bad item raises ``InvalidInputError``
        naming its position. With ``errors="none"`` bad items come back as
        ``None`` and the rest are still classified.
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError("predict_batch expects a list of texts; use predict() for a single string")
        if errors not in ("raise", "none"):
            raise ValueError("errors must be 'raise' or 'none'")
        size = self._batch_size if batch_size is None else batch_size
        if size < 1:
            raise ValueError("batch_size must be at least 1")

        items = list(texts)
        prepared: list[str | None] = []
        for index, item in enumerate(items):
            try:
                prepared.append(self._preprocessor.prepare(item))
            except InvalidInputError as exc:
                if errors == "raise":
                    raise InvalidInputError(f"Item {index}: {exc}") from exc
                prepared.append(None)

        valid = [i for i, value in enumerate(prepared) if value is not None]
        results: list[Prediction | None] = [None] * len(items)
        for start in range(0, len(valid), size):
            chunk = valid[start:start + size]
            scores = np.asarray(self._backend.score_fn([prepared[i] for i in chunk]), dtype=float)
            if scores.shape != (len(chunk), len(self._backend.labels)):
                raise ModelLoadError(
                    f"The model returned scores of shape {scores.shape}, expected {(len(chunk), len(self._backend.labels))}"
                )
            for row, index in zip(softmax(scores), chunk):
                results[index] = self._to_prediction(items[index], row)
        return results

    def _to_prediction(self, text: str, probabilities: np.ndarray) -> Prediction:
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        return Prediction(
            text=text,
            label=self._backend.labels[best],
            confidence=confidence,
            probabilities={label: float(p) for label, p in zip(self._backend.labels, probabilities)},
            is_confident=self._threshold is None or confidence >= self._threshold,
        )