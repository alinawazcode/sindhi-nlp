"""Paired comparison of two models scored on the same labeled split.

Uses the saved metrics JSON and misclassified-sentence CSV of each model, so no
model needs to be loaded and no new test run happens. McNemar's exact test looks
only at sentences where exactly one model is wrong.

    PYTHONPATH=src python -m sindhi_nlp.evaluation.compare_models \
        --a baseline_natural --b xlmr_mixed --split test
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import binomtest

from ..data.split_data import write_json
from ..utils.config import load_base_config


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def exact_mcnemar(only_a_wrong: int, only_b_wrong: int) -> float:
    """Two-sided exact p-value that the two models err equally often."""
    discordant = only_a_wrong + only_b_wrong
    if discordant == 0:
        return 1.0
    return float(binomtest(min(only_a_wrong, only_b_wrong), discordant, 0.5).pvalue)


def load_run(model: str, tag: str, split: str, metrics_dir: Path, reports_dir: Path, text_col: str) -> dict[str, Any]:
    metrics_path = metrics_dir / f"{model}_{tag}_{split}.json"
    errors_path = reports_dir / f"{model}_{tag}_{split}_misclassified.csv"
    for path in (metrics_path, errors_path):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Score {model} on the {split} split first.")
    metrics = json.loads(metrics_path.read_text("utf-8"))["metrics"]
    errors = pd.read_csv(errors_path, dtype=str, keep_default_na=False)
    n = int(metrics["num_examples"])
    expected_errors = round((1 - metrics["accuracy"]) * n)
    if len(errors) != expected_errors:
        raise ValueError(
            f"{errors_path.name} lists {len(errors)} errors but the metrics imply {expected_errors}: "
            "the error list was truncated, so a paired comparison is not possible."
        )
    if not errors[text_col].is_unique:
        raise ValueError(f"{errors_path.name} has repeated texts; cannot match errors between models")
    return {
        "model": model, "n": n, "accuracy": float(metrics["accuracy"]), "macro_f1": float(metrics["macro_f1"]),
        "errors": set(errors[text_col]),
    }


def compare_models(
    model_a: str, model_b: str, split: str = "test", tag: str = "external",
    base_config_path: str | Path = "configs/base.yaml", project_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    base = load_base_config(base_config_path)
    text_col = base["columns"]["text"]
    metrics_dir = root / base["paths"].get("metrics_dir", "artifacts/metrics")
    reports_dir = root / base["paths"].get("model_reports_dir", "artifacts/reports")

    a = load_run(model_a, tag, split, metrics_dir, reports_dir, text_col)
    b = load_run(model_b, tag, split, metrics_dir, reports_dir, text_col)
    if a["n"] != b["n"]:
        raise ValueError(f"The models were scored on different numbers of rows: {a['n']} vs {b['n']}")

    only_a_wrong = len(a["errors"] - b["errors"])   # B is right, A is wrong
    only_b_wrong = len(b["errors"] - a["errors"])   # A is right, B is wrong
    both_wrong = len(a["errors"] & b["errors"])
    result: dict[str, Any] = {
        "split": split, "dataset_tag": tag, "rows": a["n"],
        "models": {},
        "agreement": {
            "both_wrong": both_wrong,
            "only_a_wrong": only_a_wrong,
            "only_b_wrong": only_b_wrong,
            "both_right": a["n"] - both_wrong - only_a_wrong - only_b_wrong,
        },
        "mcnemar_exact_p_value": exact_mcnemar(only_a_wrong, only_b_wrong),
    }
    for key, run in (("a", a), ("b", b)):
        correct = a["n"] - len(run["errors"])
        low, high = wilson_interval(correct, a["n"])
        result["models"][key] = {
            "name": run["model"], "accuracy": run["accuracy"], "macro_f1": run["macro_f1"],
            "accuracy_95ci": [round(low, 4), round(high, 4)],
        }
    write_json(result, metrics_dir / f"compare_{model_a}_vs_{model_b}_{tag}_{split}.json")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired comparison of two scored models.")
    parser.add_argument("--a", required=True, help="First model name (folder name under artifacts/models)")
    parser.add_argument("--b", required=True, help="Second model name")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--tag", default="external")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)

    r = compare_models(args.a, args.b, args.split, args.tag, args.config, args.project_root)
    print(f"{args.a} vs {args.b} on {args.tag}/{args.split} ({r['rows']} rows)")
    for key in ("a", "b"):
        m = r["models"][key]
        print(f"  {m['name']:<22} macro-F1 {m['macro_f1']:.4f}  accuracy {m['accuracy']:.4f}  95% CI {m['accuracy_95ci']}")
    g = r["agreement"]
    print(f"  both right {g['both_right']}, both wrong {g['both_wrong']}, "
          f"only {args.a} wrong {g['only_a_wrong']}, only {args.b} wrong {g['only_b_wrong']}")
    p = r["mcnemar_exact_p_value"]
    print(f"  McNemar exact p-value: {p:.4f} -> " + ("difference is unlikely to be chance" if p < 0.05
          else "difference could plausibly be chance"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())