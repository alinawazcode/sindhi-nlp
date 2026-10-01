import pandas as pd
import pytest

from sindhi_nlp.data.load_data import load_processed_split, standardize_columns, verify_columns
from sindhi_nlp.data.validate_data import (
    DataValidationError,
    duplicate_stats,
    filter_valid_rows,
    normalize_labels,
    resolve_duplicates,
    validate_dataframe,
    validate_schema,
)
from sindhi_nlp.utils.config import ConfigError, load_base_config, merge_configs

LABELS = ["negative", "neutral", "positive"]


def test_validate_schema_missing_column():
    with pytest.raises(DataValidationError, match="Label"):
        validate_schema(pd.DataFrame({"Text": ["a"]}))


def test_normalize_labels_with_aliases():
    series = pd.Series([" Positive ", "NEG", "neutral", "bad", None])
    result = normalize_labels(series, LABELS, {"neg": "negative"})
    assert list(result[:3]) == ["positive", "negative", "neutral"]
    assert result.isna().tolist() == [False, False, False, True, True]


def test_alias_to_unknown_label_rejected():
    with pytest.raises(DataValidationError):
        normalize_labels(pd.Series(["x"]), LABELS, {"x": "mixed"})


def _messy():
    return pd.DataFrame(
        {
            "Text": ["a", "b", "b", "c", "c", "", "d"],
            "Label": ["positive", "negative", "negative", "neutral", "positive", "neutral", "weird"],
        }
    )


def test_validate_dataframe_counts():
    report = validate_dataframe(_messy(), LABELS)
    assert report.rows == 7
    assert report.missing_text == 1
    assert report.invalid_label_rows == 1
    assert report.invalid_labels == {"weird": 1}
    assert report.duplicate_rows == 2  # second "b" and second "c"
    assert report.duplicate_texts == 2
    assert report.conflicting_texts == 1  # "c"
    assert report.label_counts == {"negative": 2, "neutral": 1, "positive": 2}


def test_filter_valid_rows_drops_and_canonicalizes():
    df = pd.DataFrame({"Text": ["a", " ", "b"], "Label": ["POSITIVE", "neutral", "nope"]})
    result = filter_valid_rows(df, LABELS)
    assert result.to_dict("list") == {"Text": ["a"], "Label": ["positive"]}


def test_resolve_duplicates_drop():
    valid = filter_valid_rows(_messy(), LABELS)
    result, info = resolve_duplicates(valid, "drop")
    assert sorted(result["Text"]) == ["a", "b"]
    assert info == {"removed_conflict_rows": 2, "removed_duplicate_rows": 1}


def test_resolve_duplicates_report_keeps_everything():
    valid = filter_valid_rows(_messy(), LABELS)
    result, info = resolve_duplicates(valid, "report")
    assert len(result) == len(valid)
    assert info == {"removed_conflict_rows": 0, "removed_duplicate_rows": 0}


def test_resolve_duplicates_bad_policy():
    with pytest.raises(DataValidationError):
        resolve_duplicates(pd.DataFrame({"Text": [], "Label": []}), "keep-all")


def test_duplicate_stats_empty():
    assert duplicate_stats(pd.DataFrame({"Text": [], "Label": []})) == {
        "duplicate_rows": 0,
        "duplicate_texts": 0,
        "conflicting_texts": 0,
    }


def test_standardize_columns_is_case_insensitive():
    df = pd.DataFrame({" text ": ["a"], "SENTIMENT": ["positive"], "extra": ["x"]})
    result = standardize_columns(df, extra_label_columns=["sentiment"])
    assert list(result.columns) == ["Text", "Label"]
    with pytest.raises(DataValidationError):
        standardize_columns(pd.DataFrame({"foo": [1]}))


def test_verify_columns_and_processed_split(tmp_path):
    with pytest.raises(DataValidationError):
        verify_columns(pd.DataFrame({"Text": ["a"]}), ["Text", "Label"])
    path = tmp_path / "train.csv"
    pd.DataFrame({"Text": ["a"], "Label": ["alien"]}).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="alien"):
        load_processed_split(path, labels=LABELS)


def test_config_validation(tmp_path):
    from conftest import base_config
    import yaml

    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(base_config()), encoding="utf-8")
    assert load_base_config(good)["seed"] == 42

    bad_cfg = base_config()
    bad_cfg["data"]["split"]["test"] = 0.3
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="sum to 1"):
        load_base_config(bad)

    with pytest.raises(ConfigError, match="not found"):
        load_base_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="Missing required"):
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text("seed: 1\n", encoding="utf-8")
        load_base_config(incomplete)


def test_merge_configs_is_recursive():
    merged = merge_configs({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1}