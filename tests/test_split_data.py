import pandas as pd
import pytest

from sindhi_nlp.data.split_data import create_splits, write_splits
from sindhi_nlp.data.validate_data import DataValidationError, validate_dataframe


def _balanced_frame(rows_per_label: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Text": f"{label} sentence {index}", "Label": label}
            for label in ("negative", "neutral", "positive")
            for index in range(rows_per_label)
        ]
    )


def test_create_splits_are_stratified_disjoint_and_reproducible() -> None:
    frame = _balanced_frame()

    first = create_splits(frame, seed=7)
    second = create_splits(frame, seed=7)
    train, validation, test = first

    assert [len(split) for split in first] == [24, 3, 3]
    assert train["Label"].value_counts().to_dict() == {"negative": 8, "neutral": 8, "positive": 8}
    assert validation["Label"].value_counts().to_dict() == {"negative": 1, "neutral": 1, "positive": 1}
    assert test["Label"].value_counts().to_dict() == {"negative": 1, "neutral": 1, "positive": 1}
    assert set(train["Text"]).isdisjoint(validation["Text"])
    assert set(train["Text"]).isdisjoint(test["Text"])
    assert set(validation["Text"]).isdisjoint(test["Text"])
    assert [split.to_dict("records") for split in first] == [split.to_dict("records") for split in second]


def test_validate_dataframe_rejects_duplicate_normalized_text() -> None:
    frame = pd.DataFrame(
        {"Text": ["same sentence", "same sentence"], "Label": ["positive", "positive"]}
    )

    with pytest.raises(DataValidationError, match="duplicate normalized text rows"):
        validate_dataframe(frame)


def test_write_splits_writes_expected_artifacts(tmp_path) -> None:
    source = tmp_path / "merged.csv"
    _balanced_frame().to_csv(source, index=False)
    output_dir = tmp_path / "processed"

    report = write_splits(source, output_dir, seed=7)

    assert report["splits"]["train"]["rows"] == 24
    assert (output_dir / "train.csv").is_file()
    assert (output_dir / "validation.csv").is_file()
    assert (output_dir / "test.csv").is_file()
    assert (output_dir / "label_mapping.json").is_file()
    assert (tmp_path / "reports" / "split_report.json").is_file()
