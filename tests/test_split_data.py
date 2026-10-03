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


# ---- grouped (template-aware) splitting ----

from conftest import DAY_GROUPING, templated_frame  # noqa: E402
from sindhi_nlp.data.split_data import (  # noqa: E402
    ConfigError,
    group_overlap,
    grouped_stratified_split,
    make_group_key_fn,
    text_overlap,
)


def test_plain_split_leaks_templates_but_grouped_split_does_not():
    key_fn = make_group_key_fn(DAY_GROUPING)
    df = templated_frame()

    plain = stratified_split(df, RATIOS, seed=42)
    assert sum(group_overlap(plain, key_fn).values()) > 0  # weekday twins cross splits

    grouped, _ = grouped_stratified_split(df, RATIOS, 42, key_fn)
    assert group_overlap(grouped, key_fn) == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert text_overlap(grouped) == {"train_validation": 0, "train_test": 0, "validation_test": 0}


def test_grouped_split_sizes_stratification_and_completeness():
    df = templated_frame()
    splits, info = grouped_stratified_split(df, RATIOS, 42, make_group_key_fn(DAY_GROUPING))
    assert {k: len(v) for k, v in splits.items()} == {"train": 672, "validation": 84, "test": 84}
    for name, per_label in (("train", 224), ("validation", 28), ("test", 28)):
        assert splits[name]["Label"].value_counts().to_dict() == {
            "negative": per_label, "neutral": per_label, "positive": per_label,
        }
    assert sorted(pd.concat(splits.values())["Text"]) == sorted(df["Text"])
    assert info["groups"] == 120 and info["max_group_size"] == 7
    assert info["group_size_counts"] == {"7": 120}


def test_grouped_split_reproducible_and_seed_sensitive():
    df, key_fn = templated_frame(), make_group_key_fn(DAY_GROUPING)
    a, _ = grouped_stratified_split(df, RATIOS, 42, key_fn)
    b, _ = grouped_stratified_split(df.sample(frac=1.0, random_state=9), RATIOS, 42, key_fn)
    c, _ = grouped_stratified_split(df, RATIOS, 7, key_fn)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
    assert not a["test"].equals(c["test"])


def test_grouped_split_with_uneven_groups_never_exceeds_budgets():
    rows = [(f"{label} solo {i}", label) for label in ("negative", "neutral", "positive") for i in range(60)]
    big = templated_frame(frames_per_label=10)
    df = pd.concat([pd.DataFrame(rows, columns=["Text", "Label"]), big], ignore_index=True)
    splits, _ = grouped_stratified_split(df, RATIOS, 42, make_group_key_fn(DAY_GROUPING))
    assert len(splits["test"]) <= round(len(df) * 0.1) + 3
    assert group_overlap(splits, make_group_key_fn(DAY_GROUPING)) == {
        "train_validation": 0, "train_test": 0, "validation_test": 0,
    }


def test_group_key_function_rules():
    key = make_group_key_fn({"normalizers": [{"pattern": "Mon|Tue", "replace": "<DAY>"}], "ignore_punctuation_digits": True})
    assert key("Sale on Mon, 5 items!") == key("Sale on Tue 7 items") == "Sale on DAY items"
    with pytest.raises(ConfigError, match="Invalid regex"):
        make_group_key_fn({"normalizers": [{"pattern": "(unclosed"}]})
    with pytest.raises(ConfigError, match="pattern"):
        make_group_key_fn({"normalizers": ["Mon"]})
    with pytest.raises(ConfigError, match="no normalizers"):
        make_group_key_fn({"enabled": True})


def test_pipeline_with_grouping(make_templated_project):
    root, config = make_templated_project()
    report = run_pipeline(config, root)
    assert report["grouping"]["enabled"] is True
    assert report["grouping"]["groups"] == 120
    assert report["cross_split_group_overlap"] == {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert {k: v["rows"] for k, v in report["splits"].items()} == {"train": 672, "validation": 84, "test": 84}


def test_pipeline_grouping_is_reproducible(make_templated_project):
    root_a, config_a = make_templated_project("a")
    root_b, config_b = make_templated_project("b")
    run_pipeline(config_a, root_a)
    run_pipeline(config_b, root_b)
    for rel in ("data/processed/train.csv", "data/processed/validation.csv", "data/processed/test.csv",
                "data/reports/split_report.json"):
        assert _sha(root_a / rel) == _sha(root_b / rel), rel


def test_pipeline_without_grouping_reports_disabled(make_project):
    root, config = make_project()
    report = run_pipeline(config, root)
    assert report["grouping"] == {"enabled": False, "config": None}
    assert report["cross_split_group_overlap"] is None