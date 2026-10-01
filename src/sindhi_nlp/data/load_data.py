"""CSV loading helpers for the sentiment training pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .clean_data import clean_dataframe
from .validate_data import ValidationReport, validate_dataframe


def load_dataset(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Load a UTF-8 CSV and return its normalized Text and Label columns."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {source}")

    frame = pd.read_csv(source, encoding="utf-8")
    cleaned = clean_dataframe(frame)
    if validate:
        validate_dataframe(cleaned)
    return cleaned


def load_with_report(path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Load a CSV and return its normalized data plus validation findings."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {source}")

    frame = pd.read_csv(source, encoding="utf-8")
    cleaned = clean_dataframe(frame)
    return cleaned, validate_dataframe(cleaned, raise_on_error=False)
