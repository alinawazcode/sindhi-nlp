"""Turn predictions into tables you can inspect: all results, mistakes, and common confusions.

Everything here works on plain pandas data frames, so it needs no model and no
GPU. Build the frame with :func:`build_results_frame`, or pass the ``Prediction``
objects returned by :meth:`sindhi_nlp.SentimentPredictor.predict_batch`.

Example:
    >>> from sindhi_nlp.evaluation.error_analysis import build_results_frame, confusion_pairs
    >>> frame = build_results_frame(
    ...     texts=["a", "b", "c"],
    ...     y_true=["positive", "negative", "neutral"],
    ...     y_pred=["positive", "positive", "neutral"],
    ... )
    >>> int(frame["correct"].sum())
    2
    >>> confusion_pairs(frame).to_dict("records")
    [{'true': 'negative', 'predicted': 'positive', 'count': 1}]
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

RESULT_COLUMNS = ["text", "true", "predicted", "confidence", "correct"]


def build_results_frame(
    texts: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Combine texts, gold labels, predicted labels and (optionally) confidences in one table.

    Args:
        texts: The classified sentences.
        y_true: Gold labels, one per text.
        y_pred: Predicted labels, one per text.
        confidences: Optional confidence of each prediction. Missing values become NaN.

    Returns:
        A frame with the columns ``text``, ``true``, ``predicted``, ``confidence`` and ``correct``.

    Raises:
        ValueError: If the inputs differ in length.
    """
    n = len(texts)
    if len(y_true) != n or len(y_pred) != n or (confidences is not None and len(confidences) != n):
        raise ValueError("texts, y_true, y_pred and confidences must all have the same length")
    frame = pd.DataFrame(
        {
            "text": list(texts),
            "true": list(y_true),
            "predicted": list(y_pred),
            "confidence": list(confidences) if confidences is not None else [float("nan")] * n,
        }
    )
    frame["correct"] = frame["true"] == frame["predicted"]
    return frame[RESULT_COLUMNS]


def misclassified(frame: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Return the wrong predictions, most confident mistakes first.

    The most confident mistakes are usually the most informative: they are the
    sentences where the model is sure and wrong, which often exposes label noise
    or a missing pattern. Without confidences the original order is kept.

    Args:
        frame: A table from :func:`build_results_frame`.
        limit: Keep only this many rows. ``None`` keeps all.

    Returns:
        The rows where ``correct`` is false, with a fresh index.
    """
    wrong = frame.loc[~frame["correct"]]
    if wrong["confidence"].notna().any():
        wrong = wrong.sort_values("confidence", ascending=False, kind="stable")
    wrong = wrong.reset_index(drop=True)
    return wrong if limit is None else wrong.head(limit)


def confusion_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Count how often each (true label, predicted label) mistake happens.

    Returns:
        A frame with the columns ``true``, ``predicted`` and ``count``, largest first.
        It is empty when there are no mistakes.
    """
    wrong = frame.loc[~frame["correct"]]
    counts = (
        wrong.groupby(["true", "predicted"]).size().reset_index(name="count")
        if len(wrong)
        else pd.DataFrame(columns=["true", "predicted", "count"])
    )
    return counts.sort_values(["count", "true", "predicted"], ascending=[False, True, True]).reset_index(drop=True)
