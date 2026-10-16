"""Measure how many validation sentences have a "weekday twin" in the training split.

Sentences that differ only by a weekday word are near-copies. If many validation rows have
a copy in the training data, the validation score is inflated. The grouped split in
``sindhi_nlp.data.split_data`` is meant to bring this rate to 0.

Run from the project root:

    python scripts/audit_weekday_leakage.py
    python scripts/audit_weekday_leakage.py --train data/processed/train.csv --validation data/processed/validation.csv
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable

import pandas as pd

WEEKDAYS = ["اربع", "خميس", "اڱاري", "آچر", "جمعي", "سومر", "ڇنڇر"]


def weekday_pattern(weekdays: Iterable[str] = WEEKDAYS) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(day) for day in weekdays))


def twin_rate(train_texts: pd.Series, validation_texts: pd.Series, weekdays: Iterable[str] = WEEKDAYS) -> float:
    """Share of validation texts whose weekday-masked form also appears in the training texts."""
    pattern = weekday_pattern(weekdays)

    def mask(text: str) -> str:
        return pattern.sub("<DAY>", text)

    train_masked = set(train_texts.map(mask))
    return float(validation_texts.map(mask).isin(train_masked).mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--validation", default="data/processed/validation.csv")
    parser.add_argument("--text-column", default="Text")
    args = parser.parse_args(argv)

    train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    validation = pd.read_csv(args.validation, dtype=str, keep_default_na=False)
    rate = twin_rate(train[args.text_column], validation[args.text_column])
    print(f"validation rows whose weekday-masked twin is in train: {rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
