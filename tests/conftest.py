"""Make the local ``src/`` package importable until packaging is configured (Phase 8)."""
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LABELS = ["negative", "neutral", "positive"]


def base_config(seed: int = 42, policy: str = "drop") -> dict:
    return {
        "seed": seed,
        "labels": LABELS,
        "label_mapping": {"negative": 0, "neutral": 1, "positive": 2},
        "columns": {"text": "Text", "label": "Label"},
        "paths": {
            "raw_dir": "data/raw",
            "interim_dir": "data/interim",
            "processed_dir": "data/processed",
            "reports_dir": "data/reports",
        },
        "data": {
            "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "duplicate_policy": policy,
            "label_aliases": {},
            "cleaning": {},
        },
    }


def write_raw_files(raw_dir: Path) -> None:
    """300 unique balanced rows plus deliberately messy rows (307 rows total)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    clean = [(f"{label} جملو {i}", label) for label in LABELS for i in range(100)]
    pd.DataFrame(clean, columns=["Text", "Label"]).to_csv(raw_dir / "a.csv", index=False, encoding="utf-8")

    messy = [
        ("neutral جملو 0", "neutral"),  # exact duplicate
        ("positive جملو 1", "POSITIVE "),  # duplicate after label normalization
        ("  neutral\u200b جملو 5  ", "neutral"),  # duplicate after text cleaning
        ("negative جملو 0", "positive"),  # conflicts with a.csv (negative)
        ("   ", "neutral"),  # blank text
        ("متن", "unknown"),  # invalid label
        ("متن 2", ""),  # empty label
    ]
    pd.DataFrame(messy, columns=["text", "LABEL"]).to_csv(raw_dir / "b.csv", index=False, encoding="utf-8")


@pytest.fixture
def make_project(tmp_path):
    """Create an isolated project with messy raw data; returns (root, config_path)."""

    def _make(name: str = "proj", seed: int = 42, policy: str = "drop"):
        root = tmp_path / name
        write_raw_files(root / "data" / "raw")
        config_path = root / "base.yaml"
        config_path.write_text(
            yaml.safe_dump(base_config(seed, policy), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        return root, config_path

    return _make


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rows = [(f"{label} نمونو {i}", label) for label in LABELS for i in range(100)]
    return pd.DataFrame(rows, columns=["Text", "Label"])