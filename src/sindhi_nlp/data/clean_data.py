"""Deterministic text and label normalization used before model training."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

REQUIRED_COLUMNS = ("Text", "Label")
VALID_LABELS = frozenset({"positive", "negative", "neutral"})


def normalize_text(value: object) -> str:
    """Normalize Unicode and collapse whitespace without altering punctuation."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", text)


def normalize_label(value: object) -> str:
    """Return a lowercase label, or an empty string for a missing label."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def clean_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a dataset with the required columns."""
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {names}")

    cleaned = frame.loc[:, REQUIRED_COLUMNS].copy()
    cleaned["Text"] = cleaned["Text"].map(normalize_text)
    cleaned["Label"] = cleaned["Label"].map(normalize_label)
    return cleaned
