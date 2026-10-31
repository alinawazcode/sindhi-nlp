"""Early stopping, best-checkpoint logging, and training logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:  # Allows importing this module (and testing it) without transformers installed.
    from transformers import EarlyStoppingCallback, TrainerCallback
except ImportError:  # pragma: no cover - exercised only when transformers is missing

    class TrainerCallback:  # type: ignore[no-redef]
        """Stand-in so this module still imports without transformers."""

    class EarlyStoppingCallback:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("transformers is required for early stopping: pip install transformers")


class JsonlLogCallback(TrainerCallback):
    """Append every log, evaluation, and the final best checkpoint to a JSON-lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Append each training log entry to the JSONL log file."""
        if logs:
            self._write({"event": "log", "step": state.global_step, "epoch": state.epoch, **logs})

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Append each evaluation result to the JSONL log file."""
        if metrics:
            self._write({"event": "evaluate", "step": state.global_step, "epoch": state.epoch, **metrics})

    def on_train_end(self, args, state, control, **kwargs):
        """Write the final training state to the JSONL log file."""
        self._write(
            {
                "event": "train_end",
                "step": state.global_step,
                "best_checkpoint": getattr(state, "best_model_checkpoint", None),
                "best_metric": getattr(state, "best_metric", None),
            }
        )


def build_callbacks(early_stopping_patience: int, log_path: str | Path) -> list[Any]:
    """Create the training callbacks: a metric log and early stopping on validation macro-F1."""
    callbacks: list[Any] = [JsonlLogCallback(log_path)]
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stopping_patience)))
    return callbacks
