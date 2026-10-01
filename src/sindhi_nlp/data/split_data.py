"""Create reproducible stratified training, validation, and test datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .clean_data import VALID_LABELS
from .load_data import load_dataset
from .validate_data import ValidationReport, validate_dataframe

LABEL_MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
DEFAULT_SEED = 42


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {label: int((frame["Label"] == label).sum()) for label in sorted(VALID_LABELS)}


def create_splits(
    frame: pd.DataFrame, *, seed: int = DEFAULT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return disjoint 80/10/10 stratified splits from validated data."""
    validate_dataframe(frame)
    train, remaining = train_test_split(
        frame, test_size=0.20, stratify=frame["Label"], random_state=seed
    )
    validation, test = train_test_split(
        remaining, test_size=0.50, stratify=remaining["Label"], random_state=seed
    )
    return tuple(
        split.sample(frac=1, random_state=seed).reset_index(drop=True)
        for split in (train, validation, test)
    )


def write_splits(
    source: str | Path,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Load, validate, split, and save all training-data artifacts."""
    source_path = Path(source)
    destination = Path(output_dir)
    frame = load_dataset(source_path)
    validation_report: ValidationReport = validate_dataframe(frame)
    train, validation, test = create_splits(frame, seed=seed)

    destination.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        split.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8")
    (destination / "label_mapping.json").write_text(
        json.dumps(LABEL_MAPPING, indent=2) + "\n", encoding="utf-8"
    )

    report = {
        "source": str(source_path),
        "random_seed": seed,
        "split_method": "stratified 80/10/10",
        "input_validation": validation_report.to_dict(),
        "splits": {
            "train": {"rows": len(train), "label_counts": _label_counts(train)},
            "validation": {"rows": len(validation), "label_counts": _label_counts(validation)},
            "test": {"rows": len(test), "label_counts": _label_counts(test)},
        },
    }
    report_path = destination.parent / "reports" / "split_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Merged UTF-8 CSV with Text and Label columns.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for split CSV files.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for splitting.")
    args = parser.parse_args()

    report = write_splits(args.source, args.output_dir, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
