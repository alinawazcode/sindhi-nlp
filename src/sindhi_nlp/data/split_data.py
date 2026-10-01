"""Reproducible stratified splitting and the end-to-end data pipeline.

Run from the project root (one command rebuilds every processed file):

    PYTHONPATH=src python -m sindhi_nlp.data.split_data --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from ..utils.config import SPLIT_NAMES, load_base_config
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .clean_data import CleaningOptions, clean_series
from .load_data import file_sha256, load_raw_directory
from .validate_data import (
    DataValidationError,
    filter_valid_rows,
    resolve_duplicates,
    validate_dataframe,
)

logger = get_logger(__name__)

_PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))


def _check_ratios(ratios: Mapping[str, float]) -> None:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must have exactly the keys {SPLIT_NAMES}")
    if any(ratios[k] <= 0 for k in SPLIT_NAMES):
        raise ValueError("all split ratios must be positive")
    if not math.isclose(sum(ratios[k] for k in SPLIT_NAMES), 1.0, abs_tol=1e-6):
        raise ValueError("split ratios must sum to 1")


def split_sizes(n_rows: int, ratios: Mapping[str, float]) -> tuple[int, int, int]:
    """Integer (train, validation, test) sizes. Integers avoid float rounding surprises."""
    _check_ratios(ratios)
    n_test = int(round(n_rows * ratios["test"]))
    n_val = int(round(n_rows * ratios["validation"]))
    n_train = n_rows - n_val - n_test
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"{n_rows} rows are too few for ratios {dict(ratios)}")
    return n_train, n_val, n_test


def stratified_split(
    df: pd.DataFrame,
    ratios: Mapping[str, float],
    seed: int,
    text_col: str = "Text",
    label_col: str = "Label",
) -> dict[str, pd.DataFrame]:
    """Stratified train/validation/test split.

    Rows are sorted first, so the result depends only on the data and the
    seed, not on input file order.
    """
    df = df.sort_values([text_col, label_col], kind="mergesort").reset_index(drop=True)
    _, n_val, n_test = split_sizes(len(df), ratios)
    n_classes = df[label_col].nunique()
    if min(n_val, n_test) < n_classes:
        raise ValueError("validation/test splits must have at least one row per class")

    rest, test = train_test_split(
        df, test_size=n_test, stratify=df[label_col], random_state=seed, shuffle=True
    )
    train, val = train_test_split(
        rest, test_size=n_val, stratify=rest[label_col], random_state=seed, shuffle=True
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def text_overlap(splits: Mapping[str, pd.DataFrame], text_col: str = "Text") -> dict[str, int]:
    """Number of identical texts shared between each pair of splits."""
    sets = {name: set(frame[text_col]) for name, frame in splits.items()}
    return {f"{a}_{b}": len(sets[a] & sets[b]) for a, b in _PAIRS}


def assert_disjoint(splits: Mapping[str, pd.DataFrame], text_col: str = "Text") -> None:
    shared = {pair: n for pair, n in text_overlap(splits, text_col).items() if n}
    if shared:
        raise DataValidationError(f"Splits share identical texts: {shared}")


def count_labels(df: pd.DataFrame, labels: list[str], label_col: str = "Label") -> dict[str, int]:
    counts = df[label_col].value_counts()
    return {label: int(counts.get(label, 0)) for label in labels}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def run_pipeline(config_path: str | Path, project_root: str | Path = ".") -> dict[str, Any]:
    """Raw CSVs -> clean -> validate -> deduplicate -> split -> write outputs."""
    root = Path(project_root).resolve()
    cfg = load_base_config(config_path)

    seed = int(cfg["seed"])
    set_seed(seed)
    labels: list[str] = list(cfg["labels"])
    text_col, label_col = cfg["columns"]["text"], cfg["columns"]["label"]
    data_cfg = cfg["data"]
    ratios = {name: float(data_cfg["split"][name]) for name in SPLIT_NAMES}
    policy = data_cfg["duplicate_policy"]
    aliases = data_cfg.get("label_aliases") or {}
    options = CleaningOptions.from_config(data_cfg.get("cleaning"))

    raw_dir = root / cfg["paths"]["raw_dir"]
    interim_dir = root / cfg["paths"]["interim_dir"]
    processed_dir = root / cfg["paths"]["processed_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]

    logger.info("Loading raw data from %s", raw_dir)
    raw_df, sources = load_raw_directory(
        raw_dir,
        text_col,
        label_col,
        data_cfg.get("raw_text_columns") or (),
        data_cfg.get("raw_label_columns") or (),
    )

    logger.info("Cleaning %d rows", len(raw_df))
    df = raw_df.copy()
    df[text_col] = clean_series(df[text_col], options)

    validation = validate_dataframe(df, labels, text_col, label_col, aliases)
    for message, count in (
        ("rows with missing/empty text", validation.missing_text),
        ("rows with invalid labels", validation.invalid_label_rows),
        ("duplicate rows", validation.duplicate_rows),
        ("texts with conflicting labels", validation.conflicting_texts),
    ):
        if count:
            logger.warning("%d %s", count, message)

    df = filter_valid_rows(df, labels, text_col, label_col, aliases)
    df = df.sort_values([text_col, label_col], kind="mergesort").reset_index(drop=True)
    df, dup_info = resolve_duplicates(df, policy, text_col, label_col)
    logger.info("%d rows remain after validation (duplicate policy: %s)", len(df), policy)

    write_csv(df, interim_dir / "merged_clean.csv")

    splits = stratified_split(df, ratios, seed, text_col, label_col)
    overlap = text_overlap(splits, text_col)
    if any(overlap.values()):
        if policy == "report":
            logger.warning("Splits share identical texts (policy 'report'): %s", overlap)
        else:
            assert_disjoint(splits, text_col)

    output_hashes: dict[str, str] = {}
    for name, frame in splits.items():
        path = processed_dir / f"{name}.csv"
        write_csv(frame, path)
        output_hashes[f"{name}.csv"] = file_sha256(path)
    mapping_path = processed_dir / "label_mapping.json"
    write_json({label: int(cfg["label_mapping"][label]) for label in labels}, mapping_path)
    output_hashes["label_mapping.json"] = file_sha256(mapping_path)

    report: dict[str, Any] = {
        "seed": seed,
        "labels": labels,
        "ratios": ratios,
        "sources": sources,
        "total_rows_read": int(len(raw_df)),
        "preprocessing": {"cleaning": options.to_dict(), "label_aliases": dict(aliases)},
        "validation": validation.to_dict(),
        "duplicates": {"policy": policy, **dup_info},
        "total_rows_after_cleaning": int(len(df)),
        "label_counts_after_cleaning": count_labels(df, labels, label_col),
        "splits": {
            name: {"file": f"{name}.csv", "rows": int(len(frame)), "labels": count_labels(frame, labels, label_col)}
            for name, frame in splits.items()
        },
        "cross_split_text_overlap": overlap,
        "outputs_sha256": output_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    write_json(report, reports_dir / "split_report.json")
    logger.info("Wrote splits to %s and report to %s", processed_dir, reports_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild train/validation/test splits from raw data.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base.yaml")
    parser.add_argument("--project-root", default=".", help="Directory that data/ paths are relative to")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    report = run_pipeline(args.config, args.project_root)

    print(f"Total after cleaning: {report['total_rows_after_cleaning']:,}")
    for name in SPLIT_NAMES:
        info = report["splits"][name]
        print(f"  {name:<10} {info['rows']:>8,}  {info['labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())