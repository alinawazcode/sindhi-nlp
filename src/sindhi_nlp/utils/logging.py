"""Consistent logging configuration."""
from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_sindhi_nlp", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler._sindhi_nlp = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    root.setLevel(level.upper() if isinstance(level, str) else level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)