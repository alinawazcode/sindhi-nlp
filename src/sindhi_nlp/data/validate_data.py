"""Checks for missing text, invalid labels, duplicate text, and invalid schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from ..utils.config import DUPLICATE_POLICIES


class DataValidationError(ValueError):
    """Raised when data does not meet the expected schema or contract."""


def validate_schema(df: pd.DataFrame, text_col: str = "Text", label_col: str = "Label") -> None:
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s) {missing}; found {list(df.columns)}")


def build_label_lookup(labels: Sequence[str], aliases: Mapping[str, str] | None = None) -> dict[str, str]:
    lookup = {label.strip().lower(): label for label in labels}
    for alias, target in (aliases or {}).items():
        if target not in labels:
            raise DataValidationError(f"Alias {alias!r} points to unknown label {target!r}")
        lookup[str(alias).strip().lower()] = target
    return lookup


def normalize_label(value: Any, lookup: Mapping[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    return lookup.get(value.strip().lower())


def normalize_labels(
    series: pd.Series, labels: Sequence[str], aliases: Mapping[str, str] | None = None
) -> pd.Series:
    """Map raw label spellings to canonical labels; unknown values become ``None``."""
    lookup = build_label_lookup(labels, aliases)
    return series.map(lambda value: normalize_label(value, lookup)).astype(object)


def duplicate_stats(df: pd.DataFrame, text_col: str = "Text", label_col: str = "Label") -> dict[str, int]:
    if df.empty:
        return {"duplicate_rows": 0, "duplicate_texts": 0, "conflicting_texts": 0}
    labels_per_text = df.groupby(text_col)[label_col].nunique()
    text_counts = df[text_col].value_counts()
    return {
        "duplicate_rows": int(df.duplicated(subset=[text_col], keep="first").sum()),
        "duplicate_texts": int((text_counts > 1).sum()),
        "conflicting_texts": int((labels_per_text > 1).sum()),
    }


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    missing_text: int
    invalid_label_rows: int
    invalid_labels: dict[str, int]
    duplicate_rows: int
    duplicate_texts: int
    conflicting_texts: int
    label_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _missing_text_mask(df: pd.DataFrame, text_col: str) -> pd.Series:
    return df[text_col].map(lambda v: not isinstance(v, str) or v.strip() == "")


def validate_dataframe(
    df: pd.DataFrame,
    labels: Sequence[str],
    text_col: str = "Text",
    label_col: str = "Label",
    aliases: Mapping[str, str] | None = None,
) -> ValidationReport:
    """Describe data problems without modifying the frame."""
    validate_schema(df, text_col, label_col)
    missing = _missing_text_mask(df, text_col)
    normalized = normalize_labels(df[label_col], labels, aliases)
    invalid = normalized.isna()

    raw_invalid = df.loc[invalid, label_col].map(lambda v: v if isinstance(v, str) else repr(v))
    invalid_labels = {str(k): int(v) for k, v in raw_invalid.value_counts().sort_index().items()}

    valid = df.loc[~missing & ~invalid, [text_col]].copy()
    valid[label_col] = normalized[~missing & ~invalid]
    counts = valid[label_col].value_counts()

    return ValidationReport(
        rows=int(len(df)),
        missing_text=int(missing.sum()),
        invalid_label_rows=int(invalid.sum()),
        invalid_labels=invalid_labels,
        label_counts={label: int(counts.get(label, 0)) for label in labels},
        **duplicate_stats(valid, text_col, label_col),
    )


def filter_valid_rows(
    df: pd.DataFrame,
    labels: Sequence[str],
    text_col: str = "Text",
    label_col: str = "Label",
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Drop rows with missing text or invalid labels and canonicalize labels."""
    validate_schema(df, text_col, label_col)
    normalized = normalize_labels(df[label_col], labels, aliases)
    keep = ~_missing_text_mask(df, text_col) & normalized.notna()
    result = pd.DataFrame({text_col: df.loc[keep, text_col], label_col: normalized[keep]})
    return result.reset_index(drop=True)


def resolve_duplicates(
    df: pd.DataFrame,
    policy: str = "drop",
    text_col: str = "Text",
    label_col: str = "Label",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the duplicate policy.

    ``drop``: texts that appear with more than one label are removed entirely
    (the label is ambiguous); remaining exact duplicates keep their first row.
    ``report``: nothing is removed.
    """
    if policy not in DUPLICATE_POLICIES:
        raise DataValidationError(f"duplicate policy must be one of {DUPLICATE_POLICIES}, got {policy!r}")
    info = {"removed_conflict_rows": 0, "removed_duplicate_rows": 0}
    if policy == "report" or df.empty:
        return df.reset_index(drop=True), info

    labels_per_text = df.groupby(text_col)[label_col].nunique()
    conflicting = labels_per_text.index[labels_per_text > 1]
    is_conflict = df[text_col].isin(conflicting)
    info["removed_conflict_rows"] = int(is_conflict.sum())

    kept = df[~is_conflict]
    deduped = kept.drop_duplicates(subset=[text_col], keep="first")
    info["removed_duplicate_rows"] = int(len(kept) - len(deduped))
    return deduped.reset_index(drop=True), info