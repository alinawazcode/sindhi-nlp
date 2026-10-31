"""Score any predictor on your own labeled sentences.

This is the generic evaluator: it needs only a model (a local folder or a
Hugging Face name) and a CSV with a text column and a label column. For the
project's own split layout, with its one-time test-set guard, see
:mod:`sindhi_nlp.evaluation.evaluate_external`.

From Python:
    >>> from sindhi_nlp.evaluation.evaluate import evaluate_predictions, format_report
    >>> metrics = evaluate_predictions(
    ...     ["positive", "negative", "neutral"], ["positive", "negative", "positive"],
    ...     labels=["negative", "neutral", "positive"],
    ... )
    >>> round(metrics["accuracy"], 3)
    0.667

From the command line::

    python -m sindhi_nlp.evaluation.evaluate --model-dir alinawazmahar/sindhi-sentiment-xlmr \\
        --csv my_labeled_sentences.csv --text-column Text --label-column Label
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ..data.load_data import read_csv
from ..inference.predictor import SentimentPredictor
from ..inference.schemas import Prediction
from ..training.metrics import compute_metrics
from .error_analysis import build_results_frame, confusion_pairs, misclassified


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
    """Compute accuracy, macro and per-label precision/recall/F1 and the confusion matrix.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels, in the same order.
        labels: The label order for the confusion matrix (rows are true labels).

    Returns:
        The metrics dictionary described in :func:`sindhi_nlp.training.metrics.compute_metrics`.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    return compute_metrics(y_true, y_pred, labels)


def evaluate_predictor(
    predictor: SentimentPredictor,
    texts: Sequence[str],
    y_true: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run a predictor over labeled texts and score it.

    Texts the predictor rejects (empty after cleaning, too long) are skipped and
    counted in ``metrics["skipped"]`` instead of failing the whole run.

    Args:
        predictor: A loaded :class:`~sindhi_nlp.SentimentPredictor`.
        texts: The sentences to classify.
        y_true: Gold labels, one per text. Labels the model does not know are an error.

    Returns:
        ``(metrics, results)`` where ``results`` is the table from
        :func:`~sindhi_nlp.evaluation.error_analysis.build_results_frame`.

    Raises:
        ValueError: If lengths differ, a gold label is unknown to the model, or no text could be classified.
    """
    if len(texts) != len(y_true):
        raise ValueError(f"texts and y_true differ in length: {len(texts)} vs {len(y_true)}")
    unknown = sorted(set(y_true) - set(predictor.labels))
    if unknown:
        raise ValueError(f"Gold labels {unknown} are not labels of this model {list(predictor.labels)}")

    predictions = predictor.predict_batch(texts, errors="none")
    kept = [(i, p) for i, p in enumerate(predictions) if p is not None]
    if not kept:
        raise ValueError("None of the texts could be classified")
    kept_predictions: list[Prediction] = [p for _, p in kept]
    results = build_results_frame(
        [texts[i] for i, _ in kept],
        [y_true[i] for i, _ in kept],
        [p.label for p in kept_predictions],
        [p.confidence for p in kept_predictions],
    )
    metrics = evaluate_predictions(results["true"].tolist(), results["predicted"].tolist(), predictor.labels)
    metrics["skipped"] = len(texts) - len(kept)
    return metrics, results


def format_report(metrics: dict[str, Any], title: str = "") -> str:
    """Render metrics as a short plain-text report (accuracy, macro-F1, per-label table, confusion matrix)."""
    lines = [title] if title else []
    lines.append(
        f"{metrics['num_examples']:,} examples   accuracy {metrics['accuracy']:.4f}   macro-F1 {metrics['macro_f1']:.4f}"
    )
    if metrics.get("skipped"):
        lines.append(f"({metrics['skipped']} texts skipped because they could not be classified)")
    lines.append(f"{'label':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>9}")
    for label, row in metrics["per_label"].items():
        lines.append(f"{label:<12}{row['precision']:>10.3f}{row['recall']:>10.3f}{row['f1']:>10.3f}{row['support']:>9}")
    labels = metrics["labels"]
    lines.append("")
    lines.append("confusion matrix (rows = true, columns = predicted)")
    width = max(len(name) for name in labels) + 2
    lines.append(" " * width + "".join(f"{name:>{width}}" for name in labels))
    for name, row in zip(labels, metrics["confusion_matrix"]):
        lines.append(f"{name:<{width}}" + "".join(f"{value:>{width}}" for value in row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: score a model on a labeled CSV and print a report."""
    parser = argparse.ArgumentParser(description="Score a Sindhi sentiment model on a labeled CSV")
    parser.add_argument("--model-dir", required=True, help="Local model folder or Hugging Face name")
    parser.add_argument("--csv", required=True, help="CSV with a text column and a label column")
    parser.add_argument("--text-column", default="Text")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", help="Write the metrics (and the mistakes next to it) to this JSON path")
    parser.add_argument("--show-errors", type=int, default=0, metavar="N", help="Print the N most confident mistakes")
    args = parser.parse_args(argv)

    try:
        frame = read_csv(args.csv)
        for column in (args.text_column, args.label_column):
            if column not in frame.columns:
                raise ValueError(f"Column {column!r} not found in {args.csv}; columns are {list(frame.columns)}")
        predictor = SentimentPredictor.from_pretrained(args.model_dir, device=args.device)
        gold = frame[args.label_column].astype(str).str.strip().str.lower().tolist()
        metrics, results = evaluate_predictor(predictor, frame[args.text_column].astype(str).tolist(), gold)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_report(metrics, f"{predictor.info.name} on {Path(args.csv).name}"))
    wrong = misclassified(results)
    if args.show_errors:
        print(f"\nmost confident mistakes ({len(wrong)} in total)")
        for row in wrong.head(args.show_errors).itertuples():
            print(f"  {row.true} -> {row.predicted}  ({row.confidence:.3f})  {row.text}")
    pairs = confusion_pairs(results)
    if len(pairs):
        print("\nmost common mistakes")
        for row in pairs.head(5).itertuples():
            print(f"  {row.true} predicted as {row.predicted}: {row.count}")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        wrong.to_csv(out.with_name(out.stem + "_misclassified.csv"), index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
