"""Deterministic seeding for Python, NumPy, PyTorch, and Transformers."""

from __future__ import annotations

import importlib.util
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (when installed) PyTorch so a run can be repeated. Raises ``ValueError`` for a negative or non-integer seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        from transformers import set_seed as hf_set_seed
    except ImportError:
        return
    hf_set_seed(seed)
