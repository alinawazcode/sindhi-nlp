"""Sindhi NLP: sentiment classification for Sindhi text.

Quick start::

    from sindhi_nlp import SentimentPredictor

    predictor = SentimentPredictor.from_pretrained("artifacts/models/xlmr_mixed")
    print(predictor.predict("اڄ جو ڏينهن تمام سٺو آهي").label)

Names are imported lazily, so ``import sindhi_nlp`` stays fast and does not load
torch or scikit-learn until you actually create a predictor.
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "SentimentPredictor": "sindhi_nlp.inference.predictor",
    "Prediction": "sindhi_nlp.inference.schemas",
    "ModelInfo": "sindhi_nlp.inference.schemas",
    "InvalidInputError": "sindhi_nlp.inference.schemas",
    "ModelLoadError": "sindhi_nlp.inference.schemas",
    "ModelNotFoundError": "sindhi_nlp.inference.schemas",
}

__all__ = ["__version__", *_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module 'sindhi_nlp' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)