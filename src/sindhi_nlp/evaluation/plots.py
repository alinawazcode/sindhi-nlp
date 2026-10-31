"""Charts for results: confusion matrices and model comparisons.

Needs matplotlib, which is an optional extra: ``pip install "sindhi-nlp[plots]"``.
The functions draw on a plain ``matplotlib.figure.Figure`` and never touch the
global pyplot state, so they are safe in servers and notebooks alike. Every
function returns the figure and, when you pass ``path``, also saves it.

Example:
    >>> from sindhi_nlp.evaluation.plots import plot_confusion_matrix
    >>> fig = plot_confusion_matrix([[8, 1], [2, 9]], ["negative", "positive"])  # doctest: +SKIP
    >>> fig.savefig("confusion.png", dpi=150)  # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MISSING_MATPLOTLIB = 'Plotting needs matplotlib: pip install "sindhi-nlp[plots]"'


def _figure(width: float, height: float) -> Any:
    try:
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError(MISSING_MATPLOTLIB) from exc
    return Figure(figsize=(width, height), layout="constrained")


def _finish(fig: Any, path: str | Path | None, dpi: int) -> Any:
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=dpi)
    return fig


def plot_confusion_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    path: str | Path | None = None,
    normalize: bool = False,
    title: str = "Confusion matrix",
    dpi: int = 150,
) -> Any:
    """Draw a confusion matrix with the count in every cell.

    Args:
        matrix: Square matrix; rows are true labels and columns are predictions
            (the ``confusion_matrix`` entry of the metrics dictionary).
        labels: Class names in the same order as the rows.
        path: Where to save the image. ``None`` only returns the figure.
        normalize: Show each row as a share of its true class instead of raw counts.
        title: Chart title.
        dpi: Resolution of the saved image.

    Returns:
        The ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If the matrix is not square or does not match ``labels``.
        ImportError: If matplotlib is not installed.
    """
    data = np.asarray(matrix, dtype=float)
    n = len(labels)
    if data.shape != (n, n):
        raise ValueError(f"matrix has shape {data.shape}, expected {(n, n)} for {n} labels")
    shown = data / np.maximum(data.sum(axis=1, keepdims=True), 1) if normalize else data
    fig = _figure(1.4 * n + 2.5, 1.2 * n + 2)
    ax = fig.add_subplot()
    image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n), labels=list(labels))
    ax.set_yticks(range(n), labels=list(labels))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    threshold = shown.max() / 2 if shown.size else 0
    for i in range(n):
        for j in range(n):
            text = f"{shown[i, j]:.0%}" if normalize else f"{int(data[i, j]):,}"
            ax.text(j, i, text, ha="center", va="center", color="white" if shown[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return _finish(fig, path, dpi)


def plot_model_comparison(
    scores: Mapping[str, float],
    path: str | Path | None = None,
    metric: str = "macro-F1",
    title: str = "Model comparison",
    dpi: int = 150,
) -> Any:
    """Draw one horizontal bar per model, always starting at zero so differences are not exaggerated.

    Args:
        scores: Model name mapped to its score between 0 and 1.
        path: Where to save the image. ``None`` only returns the figure.
        metric: Name of the score, used as the axis label.
        title: Chart title.
        dpi: Resolution of the saved image.

    Returns:
        The ``matplotlib.figure.Figure``.

    Raises:
        ValueError: If ``scores`` is empty.
        ImportError: If matplotlib is not installed.
    """
    if not scores:
        raise ValueError("scores must not be empty")
    names = list(scores)
    values = [float(scores[name]) for name in names]
    fig = _figure(7, 0.7 * len(names) + 1.8)
    ax = fig.add_subplot()
    bars = ax.barh(range(len(names)), values, color="#2a6fb0")
    ax.set_yticks(range(len(names)), labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.set_title(title)
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.01, 0.93), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    return _finish(fig, path, dpi)


def plot_per_label_f1(
    per_label: Mapping[str, Mapping[str, float]],
    path: str | Path | None = None,
    title: str = "F1 per label",
    dpi: int = 150,
) -> Any:
    """Draw the F1 score of each label as a bar.

    Args:
        per_label: The ``per_label`` entry of the metrics dictionary.
        path: Where to save the image. ``None`` only returns the figure.
        title: Chart title.
        dpi: Resolution of the saved image.

    Returns:
        The ``matplotlib.figure.Figure``.
    """
    if not per_label:
        raise ValueError("per_label must not be empty")
    names = list(per_label)
    values = [float(per_label[name]["f1"]) for name in names]
    fig = _figure(1.6 * len(names) + 2, 4)
    ax = fig.add_subplot()
    bars = ax.bar(range(len(names)), values, color="#2a6fb0")
    ax.set_xticks(range(len(names)), labels=names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title(title)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, min(value + 0.02, 0.95), f"{value:.3f}", ha="center")
    return _finish(fig, path, dpi)
