import hashlib
import json

import pandas as pd
import pytest

from sindhi_nlp.data.split_data import (
    assert_disjoint,
    run_pipeline,
    split_sizes,
    stratified_split,
)
from sindhi_nlp.data.validate_data import DataValidationError

RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def test_sizes_are_exact_80_10_10(sample_df):
    assert split_sizes(150_000, RATIOS) == (120_000, 15_000, 15_000)
    splits = stratified_split(sample_df, RATIOS, seed=42)
    assert {k: len(v) for k, v in splits.items()} == {"train": 240, "validation": 30, "test": 30}


def test_splits_are_disjoint_and_complete(sample_df):
    splits = stratified_split(sample_df, RATIOS, seed=42)
    assert_disjoint(splits)
    combined = pd.concat(splits.values())
    assert sorted(combined["Text"]) == sorted(sample_df["Text"])


def test_splits_are_stratified(sample_df):
    splits = stratified_split(sample_df, RATIOS, seed=42)
    for name, expected in (("train", 80), ("validation", 10), ("test", 10)):
        assert splits[name]["Label"].value_counts().to_dict() == {
            "negative": expected,
            "neutral": expected,
            "positive": expected,
        }


def test_same_seed_reproducible_and_different_seed_differs(sample_df):
    a = stratified_split(sample_df, RATIOS, seed=42)
    b = stratified_split(sample_df, RATIOS, seed=42)
    c = stratified_split(sample_df, RATIOS, seed=7)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
    assert not a["test"].equals(c["test"])


def test_result_independent_of_input_order(sample_df):
    shuffled = sample_df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a = stratified_split(sample_df, RATIOS, seed=42)
    b = stratified_split(shuffled, RATIOS, seed=42)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_invalid_ratios_rejected(sample_df):
    with pytest.raises(ValueError):
        stratified_split(sample_df, {"train": 0.5, "validation": 0.1, "test": 0.1}, seed=1)
    with pytest.raises(ValueError):
        stratified_split(sample_df, {"train": 0.8, "test": 0.2}, seed=1)


def test_assert_disjoint_detects_leakage(sample_df):
    splits = stratified_split(sample_df, RATIOS, seed=42)
    splits["test"] = pd.concat([splits["test"], splits["train"].head(1)], ignore_index=True)
    with pytest.raises(DataValidationError, match="train_test"):
        assert_disjoint(splits)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_end_to_end(make_project):
    root, config = make_project()
    report = run_pipeline(config, root)

    assert report["total_rows_read"] == 307
    assert report["validation"]["missing_text"] == 1
    assert report["validation"]["invalid_label_rows"] == 2
    assert report["validation"]["conflicting_texts"] == 1
    assert report["duplicates"] == {"policy": "drop", "removed_conflict_rows": 2, "removed_duplicate_rows": 3}
    assert report["total_rows_after_cleaning"] == 299
    assert sum(s["rows"] for s in report["splits"].values()) == 299
    assert report["cross_split_text_overlap"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}

    processed = root / "data" / "processed"
    for name in ("train.csv", "validation.csv", "test.csv", "label_mapping.json"):
        assert (processed / name).is_file()
    assert json.loads((processed / "label_mapping.json").read_text("utf-8")) == {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    assert (root / "data" / "reports" / "split_report.json").is_file()
    assert (root / "data" / "interim" / "merged_clean.csv").is_file()

    train = pd.read_csv(processed / "train.csv")
    assert list(train.columns) == ["Text", "Label"]
    assert set(train["Label"]) <= {"negative", "neutral", "positive"}


def test_pipeline_does_not_modify_raw_files(make_project):
    root, config = make_project()
    before = {p.name: _sha(p) for p in (root / "data" / "raw").glob("*.csv")}
    run_pipeline(config, root)
    after = {p.name: _sha(p) for p in (root / "data" / "raw").glob("*.csv")}
    assert before == after


def test_pipeline_is_reproducible(make_project):
    root_a, config_a = make_project("a")
    root_b, config_b = make_project("b")
    run_pipeline(config_a, root_a)
    run_pipeline(config_b, root_b)
    for rel in (
        "data/processed/train.csv",
        "data/processed/validation.csv",
        "data/processed/test.csv",
        "data/processed/label_mapping.json",
        "data/reports/split_report.json",
    ):
        assert _sha(root_a / rel) == _sha(root_b / rel), rel


def test_different_seed_changes_split(make_project):
    root_a, config_a = make_project("a", seed=42)
    root_b, config_b = make_project("b", seed=43)
    run_pipeline(config_a, root_a)
    run_pipeline(config_b, root_b)
    assert _sha(root_a / "data/processed/test.csv") != _sha(root_b / "data/processed/test.csv")


def test_report_policy_keeps_duplicates(make_project):
    root, config = make_project(policy="report")
    report = run_pipeline(config, root)
    assert report["duplicates"] == {"policy": "report", "removed_conflict_rows": 0, "removed_duplicate_rows": 0}
    assert report["total_rows_after_cleaning"] == 304  # only invalid rows removed


def test_missing_raw_data_gives_clear_error(make_project):
    root, config = make_project()
    for path in (root / "data" / "raw").glob("*.csv"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="Place the source data"):
        run_pipeline(config, root)