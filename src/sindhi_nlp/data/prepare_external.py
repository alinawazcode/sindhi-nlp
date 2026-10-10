"""Prepare the human-labeled natural Sindhi dataset (train/validation/locked test).

Run from the project root:

    PYTHONPATH=src python -m sindhi_nlp.data.prepare_external
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from ..utils.config import SPLIT_NAMES, ConfigError, load_base_config, load_config
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .clean_data import CleaningOptions, clean_series
from .load_data import file_sha256, read_csv, standardize_columns
from .split_data import (
    count_labels,
    group_overlap,
    grouped_stratified_split,
    make_group_key_fn,
    text_overlap,
    write_csv,
    write_json,
)
from .validate_data import (
    DataValidationError,
    filter_valid_rows,
    resolve_duplicates,
    validate_dataframe,
)

logger = get_logger(__name__)

REQUIRED_KEYS = (
    "source_file", "text_column", "label_column",
    "split.train", "split.validation", "split.test", "output_dir",
)


def synthetic_overlap(
    splits: dict[str, pd.DataFrame], reference_paths: list[Path], text_col: str
) -> dict[str, Any] | None:
    """How many natural rows have a text that also appears in the synthetic data."""
    existing = [p for p in reference_paths if p.is_file()]
    if not existing:
        return None
    reference: set[str] = set()
    for path in existing:
        reference.update(read_csv(path)[text_col])
    per_split = {name: int(frame[text_col].isin(reference).sum()) for name, frame in splits.items()}
    total_rows = sum(len(f) for f in splits.values())
    return {
        "compared_files": [p.name for p in existing],
        "rows_per_split": per_split,
        "total_rows": sum(per_split.values()),
        "share_of_natural_rows": round(sum(per_split.values()) / total_rows, 6),
    }


def text_statistics(df: pd.DataFrame, labels: list[str], text_col: str, label_col: str) -> dict[str, Any]:
    words = df[text_col].str.split().str.len()
    per_label = {
        label: {
            "mean_words": round(float(words[df[label_col] == label].mean()), 3),
            "std_words": round(float(words[df[label_col] == label].std(ddof=0)), 3),
        }
        for label in labels
    }
    return {
        "words_per_text": {"min": int(words.min()), "max": int(words.max()), "mean": round(float(words.mean()), 3)},
        "words_per_label": per_label,
        "vocabulary_size": len(set(" ".join(df[text_col]).split())),
    }


def run_external_prep(
    base_config_path: str | Path = "configs/base.yaml",
    external_config_path: str | Path = "configs/external.yaml",
    project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    base = load_base_config(base_config_path)
    ext = load_config(external_config_path, required=REQUIRED_KEYS)

    seed = int(base["seed"])
    set_seed(seed)
    labels: list[str] = list(base["labels"])
    text_col, label_col = base["columns"]["text"], base["columns"]["label"]
    aliases = ext.get("label_aliases") or {}
    ratios = {name: float(ext["split"][name]) for name in SPLIT_NAMES}
    policy = ext.get("duplicate_policy", "drop")
    options = CleaningOptions.from_config((base.get("data") or {}).get("cleaning"))

    source = root / ext["source_file"]
    if not source.is_file():
        raise FileNotFoundError(f"Natural dataset not found: {source}")
    out_dir = root / ext["output_dir"]
    if source.resolve().parent == (root / base["paths"]["raw_dir"]).resolve():
        raise ConfigError("Keep the natural dataset out of data/raw/; it must not be mixed with the synthetic data")

    logger.info("Reading %s", source)
    raw = standardize_columns(
        read_csv(source), text_col, label_col,
        extra_text_columns=[ext["text_column"]], extra_label_columns=[ext["label_column"]],
        source=source.name,
    )
    df = raw.copy()
    df[text_col] = clean_series(df[text_col], options)

    validation = validate_dataframe(df, labels, text_col, label_col, aliases)
    df = filter_valid_rows(df, labels, text_col, label_col, aliases)
    df = df.sort_values([text_col, label_col], kind="mergesort").reset_index(drop=True)
    df, dup_info = resolve_duplicates(df, policy, text_col, label_col)
    logger.info("%d of %d rows kept after validation and de-duplication", len(df), len(raw))

    # Near-copies (same text apart from punctuation/digits) must stay in one split.
    key_fn = make_group_key_fn({"ignore_punctuation_digits": True})
    splits, group_info = grouped_stratified_split(df, ratios, seed, key_fn, text_col, label_col)
    group_counts = group_overlap(splits, key_fn, text_col)
    if any(group_counts.values()):
        raise DataValidationError(f"Splits share near-duplicate groups: {group_counts}")
    text_counts = text_overlap(splits, text_col)

    contamination = synthetic_overlap(
        splits, [root / p for p in ext.get("compare_with", [])], text_col
    )
    if contamination and contamination["total_rows"]:
        logger.warning(
            "%d natural rows (%.1f%%) also appear in the synthetic data",
            contamination["total_rows"], 100 * contamination["share_of_natural_rows"],
        )

    hashes: dict[str, str] = {}
    for name, frame in splits.items():
        path = out_dir / f"{name}.csv"
        write_csv(frame, path)
        hashes[f"{name}.csv"] = file_sha256(path)

    report: dict[str, Any] = {
        "seed": seed,
        "labels": labels,
        "ratios": ratios,
        "source": {"file": source.name, "rows": int(len(raw)), "sha256": file_sha256(source)},
        "preprocessing": {"cleaning": options.to_dict(), "label_aliases": dict(aliases)},
        "validation": validation.to_dict(),
        "duplicates": {"policy": policy, **dup_info},
        "total_rows_after_cleaning": int(len(df)),
        "label_counts_after_cleaning": count_labels(df, labels, label_col),
        "text_statistics": text_statistics(df, labels, text_col, label_col),
        "grouping": {"ignore_punctuation_digits": True, **group_info},
        "splits": {
            name: {"file": f"{name}.csv", "rows": int(len(frame)), "labels": count_labels(frame, labels, label_col)}
            for name, frame in splits.items()
        },
        "cross_split_text_overlap": text_counts,
        "cross_split_group_overlap": group_counts,
        "synthetic_overlap": contamination,
        "outputs_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    write_json(report, out_dir / "split_report.json")
    logger.info("Wrote splits and report to %s", out_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the human-labeled natural Sindhi dataset.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--external-config", default="configs/external.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    report = run_external_prep(args.config, args.external_config, args.project_root)

    print(f"Rows read: {report['source']['rows']:,}   kept: {report['total_rows_after_cleaning']:,}")
    print(f"Labels: {report['label_counts_after_cleaning']}")
    for name in SPLIT_NAMES:
        info = report["splits"][name]
        print(f"  {name:<10} {info['rows']:>6,}  {info['labels']}")
    stats = report["text_statistics"]
    print(f"Words per text (min/mean/max): {stats['words_per_text']['min']}/{stats['words_per_text']['mean']}/{stats['words_per_text']['max']}"
          f"   vocabulary: {stats['vocabulary_size']:,}")
    overlap = report["synthetic_overlap"]
    if overlap is not None:
        print(f"Rows also found in synthetic train/validation: {overlap['total_rows']:,} ({100 * overlap['share_of_natural_rows']:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())