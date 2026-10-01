import pandas as pd
import pytest

from sindhi_nlp.data.clean_data import clean_dataframe, normalize_text


def test_normalize_text_normalizes_unicode_and_whitespace() -> None:
    assert normalize_text("  سلام\t\nدنيا  ") == "سلام دنيا"


def test_clean_dataframe_normalizes_labels() -> None:
    frame = pd.DataFrame({"Text": [" متن "], "Label": [" Positive "]})

    cleaned = clean_dataframe(frame)

    assert cleaned.to_dict("records") == [{"Text": "متن", "Label": "positive"}]


def test_clean_dataframe_requires_text_and_label_columns() -> None:
    with pytest.raises(ValueError, match="Label"):
        clean_dataframe(pd.DataFrame({"Text": ["متن"]}))
