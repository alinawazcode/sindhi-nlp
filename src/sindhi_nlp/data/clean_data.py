"""Unicode and whitespace normalization.

The same function must be used in training and at inference time, so it is
deterministic, idempotent, and free of I/O.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

# Soft hyphen, zero-width space/joiner, bidi marks and isolates, word joiner, BOM.
INVISIBLE_CHARS = "\u00ad\u200b\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2060\u2066\u2067\u2068\u2069\ufeff"
ZWNJ = "\u200c"
TATWEEL = "\u0640"

_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")
# C0/C1 control characters that are not whitespace (\s already covers the rest).
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1b\x7f-\x84\x86-\x9f]")
_WHITESPACE_RE = re.compile(r"\s+")
_UNICODE_FORMS = {"NFC", "NFD", "NFKC", "NFKD"}


@dataclass(frozen=True)
class CleaningOptions:
    unicode_form: str | None = "NFKC"
    remove_invisible: bool = True
    remove_zwnj: bool = False
    remove_tatweel: bool = True
    strip_diacritics: bool = False
    char_map: Mapping[str, str] = field(default_factory=dict)
    _table: dict[int, str | None] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.unicode_form in ("none", ""):
            object.__setattr__(self, "unicode_form", None)
        if self.unicode_form is not None and self.unicode_form not in _UNICODE_FORMS:
            raise ValueError(f"unicode_form must be one of {sorted(_UNICODE_FORMS)} or null")
        for source, target in self.char_map.items():
            if len(source) != 1:
                raise ValueError(f"char_map keys must be single characters, got {source!r}")
            if not isinstance(target, str):
                raise ValueError(f"char_map values must be strings, got {target!r}")

        table: dict[int, str | None] = {}
        if self.remove_invisible:
            table.update({ord(c): None for c in INVISIBLE_CHARS})
        if self.remove_zwnj:
            table[ord(ZWNJ)] = None
        if self.remove_tatweel:
            table[ord(TATWEEL)] = None
        table.update({ord(k): v for k, v in self.char_map.items()})
        object.__setattr__(self, "_table", table)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "CleaningOptions":
        config = dict(config or {})
        known = {"unicode_form", "remove_invisible", "remove_zwnj", "remove_tatweel", "strip_diacritics", "char_map"}
        unknown = set(config) - known
        if unknown:
            raise ValueError(f"Unknown cleaning option(s): {sorted(unknown)}")
        return cls(**config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unicode_form": self.unicode_form,
            "remove_invisible": self.remove_invisible,
            "remove_zwnj": self.remove_zwnj,
            "remove_tatweel": self.remove_tatweel,
            "strip_diacritics": self.strip_diacritics,
            "char_map": dict(self.char_map),
        }


DEFAULT_OPTIONS = CleaningOptions()


def clean_text(text: Any, options: CleaningOptions = DEFAULT_OPTIONS) -> str:
    """Normalize one text. Non-string input (None, NaN, numbers) becomes ``""``."""
    if not isinstance(text, str):
        return ""
    if options.unicode_form:
        text = unicodedata.normalize(options.unicode_form, text)
    if options._table:
        text = text.translate(options._table)
        # Removing characters can expose new compositions, so normalize again.
        if options.unicode_form:
            text = unicodedata.normalize(options.unicode_form, text)
    if options.strip_diacritics:
        text = _DIACRITICS_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_series(series: pd.Series, options: CleaningOptions = DEFAULT_OPTIONS) -> pd.Series:
    return series.map(lambda value: clean_text(value, options)).astype(object)