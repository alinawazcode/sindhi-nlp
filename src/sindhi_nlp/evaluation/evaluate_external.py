"""Score a saved model on a labeled CSV (for example the natural Sindhi splits).

Validation can be scored as often as you like. The test split is a one-time,
final evaluation: it needs ``--final`` and refuses to run a second time for the
same model unless you pass ``--overwrite``.

    PYTHONPATH=src python -m sindhi_nlp.evaluation.evaluate_external \
        --model-dir artifacts/models/baseline_tfidf_linearsvc --split validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..data.clean_data import CleaningOptions, clean_series
from ..data.load_data import load_processed_split
from ..data.split_data import write_csv, write_json
from ..training.baseline import error_analysis
from ..training.metrics import compute_metrics
from ..utils.config import load_base_config
from ..utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class FinalEvaluationError(RuntimeError):
    """Raised when the one-time test evaluation is misused."""


def hf_predict(model_dir: Path, texts: list[str], max_length: int = 128, batch_size: int = 64) -> np.ndarray:
    """Predict label names with a saved Hugging Face sequence-classification model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    id2label = {int(i): name for i, name in model.config.id2label.items()}
    predictions: list[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], truncation=True, max_length=max_length,
                              padding=True, return_tensors="pt").to(device)
            predictions.extend(id2label[i] for i in model(**batch).logits.argmax(dim=-1).tolist())
    return np.array(predictions, dtype=object)


def _is_hf_model(model_dir: Path) -> bool:
    return any((model_dir / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))


def _is_test(split: str, data_path: Path) -> bool:
    return split == "test" or data_path.stem.lower() == "test"


def evaluate_model(
    model_dir: str | Path,
    data_path: str | Path,
    split: str = "validation",
    tag: str = "external",
    base_config_path: str | Path = "configs/base.yaml",
    project_root: str | Path = ".",
    final: bool = False,
    overwrite: bool = False,
    max_error_examples: int = 200,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    model_dir, data_path = root / Path(model_dir), root / Path(data_path)
    model_name = model_dir.name
    base = load_base_config(base_config_path)
    labels: list[str] = list(base["labels"])
    text_col, label_col = base["columns"]["text"], base["columns"]["label"]
    metrics_dir = root / base["paths"].get("metrics_dir", "artifacts/metrics")
    reports_dir = root / base["paths"].get("model_reports_dir", "artifacts/reports")
    result_path = metrics_dir / f"{model_name}_{tag}_{split}.json"

    if _is_test(split, data_path):
        if not final:
            raise FinalEvaluationError("The test split is a one-time final evaluation. Pass --final to confirm.")
        if result_path.exists() and not overwrite:
            raise FinalEvaluationError(
                f"{result_path.name} already exists: the test split was already scored for this model. "
                "Pass --overwrite only if you are certain."
            )

    model_file = model_dir / "model.joblib"
    model = None
    if model_file.is_file():
        model = joblib.load(model_file)
    elif not _is_hf_model(model_dir):
        raise FileNotFoundError(
            f"No model found in {model_dir}: expected model.joblib (baseline) or model.safetensors (XLM-R)."
        )

    df = load_processed_split(data_path, text_col, label_col, labels)
    options = CleaningOptions.from_config((base.get("data") or {}).get("cleaning"))
    df[text_col] = clean_series(df[text_col], options)

    if model is not None:
        predictions = model.predict(df[text_col])
    else:
        max_length = 128
        training_config = model_dir / "training_config.json"
        if training_config.is_file():
            max_length = int(json.loads(training_config.read_text("utf-8"))["config"]["tokenizer"]["max_length"])
        predictions = hf_predict(model_dir, df[text_col].tolist(), max_length=max_length)
    metrics = compute_metrics(df[label_col], predictions, labels)

    if "svc" in getattr(model, "named_steps", {}):
        errors, confusions = error_analysis(model, df, predictions, max_error_examples, text_col, label_col)
    else:
        wrong = df[label_col].to_numpy() != predictions
        errors = pd.DataFrame({
            text_col: df[text_col].to_numpy()[wrong],
            "true_label": df[label_col].to_numpy()[wrong],
            "predicted_label": predictions[wrong],
        }).head(max_error_examples)
        confusions = []

    report = {
        "model": model_name,
        "dataset": {"tag": tag, "split": split, "file": data_path.name, "rows": int(len(df))},
        "metrics": metrics,
        "top_confusions": confusions,
    }
    write_json(report, result_path)
    write_csv(errors, reports_dir / f"{model_name}_{tag}_{split}_misclassified.csv")
    logger.info("Saved %s", result_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on a labeled CSV.")
    parser.add_argument("--model-dir", default="artifacts/models/baseline_tfidf_linearsvc")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--data", default=None, help="CSV path (default: data/external/processed/<split>.csv)")
    parser.add_argument("--tag", default="external", help="Dataset label used in output file names")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--final", action="store_true", help="Confirm the one-time test evaluation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    data = args.data or f"data/external/processed/{args.split}.csv"
    report = evaluate_model(
        args.model_dir, data, args.split, args.tag, args.config, args.project_root, args.final, args.overwrite
    )
    m = report["metrics"]
    print(f"{report['model']} on {args.tag}/{args.split} ({m['num_examples']:,} rows)")
    print(f"  macro-F1 {m['macro_f1']:.4f}   accuracy {m['accuracy']:.4f}")
    for label in m["labels"]:
        s = m["per_label"][label]
        print(f"  {label:<10} P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}  (n={s['support']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())