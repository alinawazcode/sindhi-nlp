"""Text preparation applied before tokenization at prediction time.

This reuses ``clean_text`` from the training pipeline, so training and
inference see identically normalized text. The cleaning options a model was
trained with live in ``preprocessing.json`` next to the model; if that file is
missing the project defaults are used (they match ``configs/base.yaml``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.clean_data import CleaningOptions, clean_text
from .schemas import InvalidInputError, ModelLoadError

PREPROCESSING_FILE = "preprocessing.json"
DEFAULT_MAX_CHARS = 10_000


class TextPreprocessor:
    def __init__(
        self,
        options: CleaningOptions | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
        source: str = "defaults",
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        self.options = options or CleaningOptions()
        self.max_chars = max_chars
        self.source = source

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, max_chars: int = DEFAULT_MAX_CHARS) -> "TextPreprocessor":
        path = Path(model_dir) / PREPROCESSING_FILE
        if not path.is_file():
            return cls(max_chars=max_chars, source="defaults (no preprocessing.json found)")
        try:
            cleaning = json.loads(path.read_text("utf-8")).get("cleaning")
            options = CleaningOptions.from_config(cleaning)
        except (ValueError, OSError) as exc:
            raise ModelLoadError(f"Invalid {path}: {exc}") from exc
        return cls(options, max_chars, source=PREPROCESSING_FILE)

    def prepare(self, text: Any) -> str:
        """Return cleaned text, or raise ``InvalidInputError`` if it cannot be classified."""
        if not isinstance(text, str):
            raise InvalidInputError(f"Text must be a string, got {type(text).__name__}")
        if len(text) > self.max_chars:
            raise InvalidInputError(f"Text has {len(text):,} characters; the limit is {self.max_chars:,}")
        cleaned = clean_text(text, self.options)
        if not cleaned:
            raise InvalidInputError("Text is empty after cleaning")
        return cleaned


def write_preprocessing_config(model_dir: str | Path, options: CleaningOptions, source: str = "") -> Path:
    """Save the cleaning options next to a model so inference can reproduce them."""
    path = Path(model_dir) / PREPROCESSING_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cleaning": options.to_dict(), "source": source}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    return path