"""Accuracy, precision, recall, macro-F1, and per-label metrics."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> dict[str, Any]:
    """Classification metrics over a fixed label order.

    ``confusion_matrix`` rows are true labels and columns are predictions,
    both in the order of ``labels``. Macro averages weight every label equally.
    """
    labels = list(labels)
    y_true, y_pred = list(y_true), list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred differ in length: {len(y_true)} vs {len(y_pred)}")
    if not y_true:
        raise ValueError("Cannot compute metrics on zero examples")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "num_examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "per_label": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "labels": [str(label) for label in labels],
        "confusion_matrix": matrix.tolist(),
    }


def make_hf_compute_metrics(labels_by_id: Sequence[str]) -> Callable[[Any], dict[str, float]]:
    """Build a ``compute_metrics`` callback for the Hugging Face Trainer.

    ``labels_by_id[i]`` must be the name of class id ``i``. Returned keys are
    flat floats so the Trainer can log them and select the best checkpoint
    with ``metric_for_best_model="macro_f1"``.
    """
    names = list(labels_by_id)
    ids = list(range(len(names)))

    def _compute(eval_pred: Any) -> dict[str, float]:
        logits = getattr(eval_pred, "predictions", None)
        label_ids = getattr(eval_pred, "label_ids", None)
        if logits is None:  # a plain (predictions, label_ids) tuple
            logits, label_ids = eval_pred[0], eval_pred[1]
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        result = compute_metrics(np.asarray(label_ids).tolist(), predictions.tolist(), ids)
        flat = {key: result[key] for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall")}
        for i, name in enumerate(names):
            flat[f"f1_{name}"] = result["per_label"][str(i)]["f1"]
        return flat

    return _compute