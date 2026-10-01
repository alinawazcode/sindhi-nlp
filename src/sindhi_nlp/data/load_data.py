"""Read CSV files and verify the expected columns."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .validate_data import DataValidationError


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV as plain strings (no NaN coercion, BOM tolerated)."""
    path = Path(path)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"{path.name} is not valid UTF-8: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def verify_columns(df: pd.DataFrame, expected: Iterable[str], source: str = "data") -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"{source}: missing column(s) {missing}; found {list(df.columns)}")


def _find_column(columns: Sequence[str], candidates: Iterable[str]) -> str | None:
    lowered = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        match = lowered.get(candidate.strip().lower())
        if match is not None:
            return match
    return None


def standardize_columns(
    df: pd.DataFrame,
    text_col: str = "Text",
    label_col: str = "Label",
    extra_text_columns: Iterable[str] = (),
    extra_label_columns: Iterable[str] = (),
    source: str = "data",
) -> pd.DataFrame:
    """Rename accepted header variants to the canonical column names."""
    text_found = _find_column(df.columns, [text_col, *extra_text_columns])
    label_found = _find_column(df.columns, [label_col, *extra_label_columns])
    if text_found is None or label_found is None:
        verify_columns(df, [text_col, label_col], source)
    return df.rename(columns={text_found: text_col, label_found: label_col})[[text_col, label_col]]


def load_raw_directory(
    raw_dir: str | Path,
    text_col: str = "Text",
    label_col: str = "Label",
    extra_text_columns: Iterable[str] = (),
    extra_label_columns: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load and concatenate every ``*.csv`` in ``raw_dir`` (sorted by name).

    Returns the merged frame and per-file source details (name, rows, sha256).
    Raw files are only read, never modified.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. Place the source data there first.")

    frames, sources = [], []
    for path in files:
        df = standardize_columns(
            read_csv(path), text_col, label_col, extra_text_columns, extra_label_columns, source=path.name
        )
        frames.append(df)
        sources.append({"file": path.name, "rows": int(len(df)), "sha256": file_sha256(path)})
    return pd.concat(frames, ignore_index=True), sources


def load_processed_split(
    path: str | Path,
    text_col: str = "Text",
    label_col: str = "Label",
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load one of train/validation/test.csv and verify its schema and labels."""
    path = Path(path)
    df = read_csv(path)
    verify_columns(df, [text_col, label_col], path.name)
    if labels is not None:
        bad = sorted(set(df[label_col]) - set(labels))
        if bad:
            raise DataValidationError(f"{path.name}: unexpected label(s) {bad}")
    return df[[text_col, label_col]]