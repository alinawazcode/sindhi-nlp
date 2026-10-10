"""TF-IDF + LinearSVC baseline (Phase 3).

Trains on ``train.csv`` only and evaluates on ``validation.csv``. It never
reads ``test.csv``; that file is reserved for the one-time final evaluation.

Run from the project root:

    PYTHONPATH=src python -m sindhi_nlp.training.baseline
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..data.load_data import file_sha256, load_processed_split
from ..data.split_data import write_csv, write_json
from ..utils.config import load_base_config, load_config
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from .metrics import compute_metrics

logger = get_logger(__name__)

REQUIRED_BASELINE_KEYS = ("model_name", "tfidf.analyzer", "tfidf.ngram_range", "svc.C")


def build_pipeline(baseline_cfg: dict[str, Any], seed: int) -> Pipeline:
    tfidf = dict(baseline_cfg["tfidf"])
    tfidf["ngram_range"] = tuple(tfidf["ngram_range"])
    svc = dict(baseline_cfg["svc"])
    svc.setdefault("dual", "auto")
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("svc", LinearSVC(random_state=seed, **svc)),
        ]
    )


def top_features(pipeline: Pipeline, top_n: int) -> dict[str, list[dict[str, float]]]:
    """Highest-weighted terms per class, useful for spotting label artifacts."""
    vectorizer, svc = pipeline.named_steps["tfidf"], pipeline.named_steps["svc"]
    names = vectorizer.get_feature_names_out()
    result: dict[str, list[dict[str, float]]] = {}
    for row, label in zip(svc.coef_, svc.classes_):
        best = np.argsort(row)[::-1][:top_n]
        result[str(label)] = [{"term": str(names[i]), "weight": round(float(row[i]), 4)} for i in best]
    return result


def error_analysis(
    pipeline: Pipeline, df: pd.DataFrame, predictions: np.ndarray, max_examples: int,
    text_col: str, label_col: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Most confident mistakes and the most frequent (true -> predicted) confusions."""
    svc = pipeline.named_steps["svc"]
    classes = list(svc.classes_)
    scores = pipeline.decision_function(df[text_col])
    index = {label: i for i, label in enumerate(classes)}
    rows = np.arange(len(df))
    true_idx = df[label_col].map(index).to_numpy()
    pred_idx = np.array([index[p] for p in predictions])
    margin = scores[rows, pred_idx] - scores[rows, true_idx]

    wrong = df[label_col].to_numpy() != predictions
    errors = pd.DataFrame(
        {
            text_col: df[text_col].to_numpy()[wrong],
            "true_label": df[label_col].to_numpy()[wrong],
            "predicted_label": predictions[wrong],
            "margin": np.round(margin[wrong], 4),
        }
    )
    errors = errors.sort_values(["margin", text_col], ascending=[False, True], kind="mergesort")

    pairs = (
        errors.groupby(["true_label", "predicted_label"]).size().sort_values(ascending=False).reset_index(name="count")
    )
    confusions = [
        {"true": r.true_label, "predicted": r.predicted_label, "count": int(r["count"])}
        for _, r in pairs.iterrows()
    ]
    return errors.head(max_examples).reset_index(drop=True), confusions


def format_summary(report: dict[str, Any]) -> str:
    m = report["validation_metrics"]
    labels = m["labels"]
    lines = [
        f"# Baseline report: {report['model_name']}",
        "",
        "Trained on `train.csv`, evaluated on `validation.csv`. The test split was not used.",
        "",
        "## Validation results",
        "",
        f"- Macro-F1: **{m['macro_f1']:.4f}**",
        f"- Accuracy: {m['accuracy']:.4f}",
        f"- Macro precision / recall: {m['macro_precision']:.4f} / {m['macro_recall']:.4f}",
        f"- Train time: {report['timing']['fit_seconds']:.1f}s, validation prediction: {report['timing']['predict_seconds']:.2f}s",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        s = m["per_label"][label]
        lines.append(f"| {label} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} | {s['support']:,} |")

    lines += ["", "## Confusion matrix (rows = true, columns = predicted)", "", "| true \\ predicted | " + " | ".join(labels) + " |", "| --- |" + " ---: |" * len(labels)]
    for label, row in zip(labels, m["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(f"{v:,}" for v in row) + " |")

    lines += ["", "## Most common mistakes", ""]
    for c in report["top_confusions"][:6]:
        lines.append(f"- {c['true']} predicted as {c['predicted']}: {c['count']:,}")

    lines += ["", "## Highest-weighted terms per class", ""]
    for label, terms in report["top_features"].items():
        lines.append(f"- **{label}**: " + ", ".join(t["term"] for t in terms))

    lines += ["", "## Setup", "", f"- Seed: {report['seed']}", f"- TF-IDF: `{json.dumps(report['config']['tfidf'], ensure_ascii=False)}`", f"- LinearSVC: `{json.dumps(report['config']['svc'])}`", f"- Train rows: {report['data']['train_rows']:,}, validation rows: {report['data']['validation_rows']:,}", ""]
    return "\n".join(lines)


def run_baseline(
    base_config_path: str | Path = "configs/base.yaml",
    baseline_config_path: str | Path = "configs/baseline.yaml",
    project_root: str | Path = ".",
    data_dir: str | Path | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Train on ``<data_dir>/train.csv`` and evaluate on ``<data_dir>/validation.csv``.

    ``data_dir`` defaults to the synthetic processed folder from the base config;
    pass ``data/external/processed`` for the natural dataset, together with a new
    ``model_name`` so the earlier baseline is not overwritten.
    """
    root = Path(project_root).resolve()
    base_cfg = load_base_config(base_config_path)
    cfg = load_config(baseline_config_path, required=REQUIRED_BASELINE_KEYS)

    seed = int(base_cfg["seed"])
    set_seed(seed)
    labels: list[str] = list(base_cfg["labels"])
    text_col, label_col = base_cfg["columns"]["text"], base_cfg["columns"]["label"]
    paths = base_cfg["paths"]
    processed = root / (data_dir if data_dir is not None else paths["processed_dir"])
    models_dir = root / paths.get("models_dir", "artifacts/models")
    metrics_dir = root / paths.get("metrics_dir", "artifacts/metrics")
    reports_dir = root / paths.get("model_reports_dir", "artifacts/reports")

    train_path, val_path = processed / "train.csv", processed / "validation.csv"
    train = load_processed_split(train_path, text_col, label_col, labels)
    val = load_processed_split(val_path, text_col, label_col, labels)
    logger.info("Loaded %d train and %d validation rows", len(train), len(val))

    pipeline = build_pipeline(cfg, seed)
    start = time.perf_counter()
    pipeline.fit(train[text_col], train[label_col])
    fit_seconds = time.perf_counter() - start
    logger.info("Trained in %.1fs", fit_seconds)

    start = time.perf_counter()
    predictions = pipeline.predict(val[text_col])
    predict_seconds = time.perf_counter() - start

    metrics = compute_metrics(val[label_col], predictions, labels)
    logger.info("Validation macro-F1 %.4f, accuracy %.4f", metrics["macro_f1"], metrics["accuracy"])

    analysis = cfg.get("analysis") or {}
    errors, confusions = error_analysis(
        pipeline, val, predictions, int(analysis.get("max_error_examples", 200)), text_col, label_col
    )
    features = top_features(pipeline, int(analysis.get("top_features_per_class", 15)))

    model_name = model_name or cfg["model_name"]
    config_snapshot = {k: cfg[k] for k in ("tfidf", "svc") if k in cfg}
    report: dict[str, Any] = {
        "model_name": model_name,
        "data_dir": str(Path(data_dir) if data_dir is not None else paths["processed_dir"]),
        "seed": seed,
        "config": config_snapshot,
        "data": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
            "train_sha256": file_sha256(train_path),
            "validation_sha256": file_sha256(val_path),
        },
        "validation_metrics": metrics,
        "top_confusions": confusions,
        "top_features": features,
        "timing": {"fit_seconds": round(fit_seconds, 3), "predict_seconds": round(predict_seconds, 3)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }

    # Everything needed to reload and audit the model lives in one folder.
    model_dir = models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    write_json({label: int(base_cfg["label_mapping"][label]) for label in labels}, model_dir / "label_mapping.json")
    write_json({"seed": seed, **config_snapshot, "labels": labels}, model_dir / "config.json")
    write_json({"validation": metrics, "data": report["data"], "seed": seed}, model_dir / "metrics.json")

    write_json(report, metrics_dir / f"{model_name}_validation.json")
    write_csv(errors, reports_dir / f"{model_name}_misclassified_validation.csv")
    summary_path = reports_dir / f"{model_name}_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(format_summary(report), encoding="utf-8", newline="\n")
    logger.info("Saved model to %s and reports to %s", model_dir, reports_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and validate the TF-IDF + LinearSVC baseline.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-dir", default=None, help="Folder with train.csv and validation.csv (default: synthetic)")
    parser.add_argument("--model-name", default=None, help="Output name, e.g. baseline_natural")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    report = run_baseline(args.config, args.baseline_config, args.project_root, args.data_dir, args.model_name)

    m = report["validation_metrics"]
    print(f"Validation macro-F1: {m['macro_f1']:.4f}   accuracy: {m['accuracy']:.4f}")
    for label in m["labels"]:
        print(f"  {label:<10} F1 {m['per_label'][label]['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())