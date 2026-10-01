import pytest

from sindhi_nlp.data.clean_data import CleaningOptions, clean_series, clean_text


def test_nfkc_converts_presentation_forms():
    assert clean_text("\ufe8d") == "\u0627"


def test_whitespace_is_collapsed_and_stripped():
    assert clean_text("  a \t b\n\u00a0c  ") == "a b c"


def test_invisible_characters_removed_but_zwnj_kept_by_default():
    assert clean_text("ab\u200bc\ufeffd") == "abcd"
    assert clean_text("a\u200cb") == "a\u200cb"
    assert clean_text("a\u200cb", CleaningOptions(remove_zwnj=True)) == "ab"


def test_tatweel_removed():
    assert clean_text("ک\u0640تاب") == "کتاب"
    assert clean_text("ک\u0640تاب", CleaningOptions(remove_tatweel=False)) == "ک\u0640تاب"


def test_control_characters_removed():
    assert clean_text("a\x00b\x07c") == "abc"


@pytest.mark.parametrize("value", [None, float("nan"), 12, b"bytes", ["x"]])
def test_non_string_becomes_empty(value):
    assert clean_text(value) == ""


def test_empty_after_cleaning():
    assert clean_text(" \u200b\u200f ") == ""


def test_char_map_applied():
    options = CleaningOptions(char_map={"\u0643": "\u06a9"})
    assert clean_text("\u0643تاب", options) == "\u06a9تاب"


def test_strip_diacritics_optional():
    text = "ك\u064eتاب"
    assert "\u064e" in clean_text(text)
    assert "\u064e" not in clean_text(text, CleaningOptions(strip_diacritics=True))


@pytest.mark.parametrize(
    "text",
    ["  هي  جملو  ", "ک\u0640تاب\u200b", "\ufe8d\ufe8e", "plain ascii", "a\u200cb\u00a0c"],
)
def test_cleaning_is_idempotent(text):
    once = clean_text(text)
    assert clean_text(once) == once


def test_clean_series_returns_strings():
    import pandas as pd

    result = clean_series(pd.Series([" a ", None, "b\u200b"]))
    assert list(result) == ["a", "", "b"]


def test_from_config_rejects_unknown_and_bad_options():
    with pytest.raises(ValueError, match="Unknown cleaning option"):
        CleaningOptions.from_config({"nope": True})
    with pytest.raises(ValueError, match="single characters"):
        CleaningOptions(char_map={"ab": "c"})
    with pytest.raises(ValueError, match="unicode_form"):
        CleaningOptions(unicode_form="BOGUS")
    assert CleaningOptions.from_config({"unicode_form": "none"}).unicode_form is None
    assert CleaningOptions.from_config(None) == CleaningOptions()