"""Validation rules that prevent flawed datasets from entering training."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .clean_data import REQUIRED_COLUMNS, VALID_LABELS


class DataValidationError(ValueError):
    """Raised when a dataset fails the training-data quality checks."""


@dataclass(frozen=True)
class ValidationReport:
    """A compact, serializable summary of a dataset validation run."""

    total_rows: int
    empty_text_rows: int
    invalid_label_rows: int
    duplicate_text_rows: int
    label_counts: dict[str, int]

    @property
    def is_valid(self) -> bool:
        return (
            self.total_rows > 0
            and self.empty_text_rows == 0
            and self.invalid_label_rows == 0
            and self.duplicate_text_rows == 0
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_dataframe(frame: pd.DataFrame, *, raise_on_error: bool = True) -> ValidationReport:
    """Validate normalized Text/Label data and optionally raise on failures."""
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"Dataset is missing required columns: {names}")

    empty_text_rows = int(frame["Text"].eq("").sum())
    invalid_label_rows = int((~frame["Label"].isin(VALID_LABELS)).sum())
    duplicate_mask = frame["Text"].duplicated(keep=False)
    duplicate_text_rows = int(duplicate_mask.sum())
    label_counts = {
        label: int(count)
        for label, count in frame["Label"].value_counts().sort_index().items()
    }
    report = ValidationReport(
        total_rows=len(frame),
        empty_text_rows=empty_text_rows,
        invalid_label_rows=invalid_label_rows,
        duplicate_text_rows=duplicate_text_rows,
        label_counts=label_counts,
    )

    if raise_on_error and not report.is_valid:
        problems: list[str] = []
        if report.total_rows == 0:
            problems.append("dataset has no rows")
        if report.empty_text_rows:
            problems.append(f"{report.empty_text_rows} empty text rows")
        if report.invalid_label_rows:
            problems.append(f"{report.invalid_label_rows} invalid label rows")
        if report.duplicate_text_rows:
            problems.append(f"{report.duplicate_text_rows} duplicate normalized text rows")
        raise DataValidationError("Dataset validation failed: " + "; ".join(problems))

    return report
