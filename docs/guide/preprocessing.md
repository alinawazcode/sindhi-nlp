# Text preprocessing

Models are sensitive to invisible differences in Unicode text. Sindhi is written in an Arabic-based script with many look-alike characters, so the project applies one deterministic cleaning function, `clean_text`, in both training and prediction. A model that was trained on cleaned text must see cleaned text.

## What the default cleaning does

In order:

1. **Unicode normalization** to `NFKC` (compatibility forms such as presentation ligatures become their plain letters).
2. **Remove invisible characters**: zero-width space and joiner, soft hyphen, byte-order mark, bidirectional marks and isolates, word joiner.
3. **Remove tatweel** (kashida, U+0640), the decorative stretch between letters.
4. **Normalize again**, because removing characters can expose new compositions.
5. **Remove control characters** that are not whitespace.
6. **Collapse whitespace**: runs of spaces, tabs and newlines become one space; the ends are trimmed.

Not done by default: removing ZWNJ (U+200C, which is meaningful in some Sindhi words), removing diacritics, changing digits, changing punctuation, or mapping letter variants. Those are explicit options.

Real examples:

```python
from sindhi_nlp.data.clean_data import clean_text, CleaningOptions

clean_text("سٺو​  ڏينهنـ ")  # 'سٺو ڏينهن'    zero-width space, tatweel and extra spaces removed
clean_text("  a\t\nb  ")  # 'a b'
clean_text(None)  # ''             non-strings become an empty string
clean_text("١٢٣")  # '١٢٣'          digits are left alone
```

## Options

`CleaningOptions` is a frozen dataclass:

| Option | Default | Effect |
| --- | --- | --- |
| `unicode_form` | `"NFKC"` | `"NFC"`, `"NFD"`, `"NFKC"`, `"NFKD"`, or `None` for no normalization |
| `remove_invisible` | `True` | drop zero-width and bidirectional control characters |
| `remove_zwnj` | `False` | also drop ZWNJ (U+200C) |
| `remove_tatweel` | `True` | drop kashida (U+0640) |
| `strip_diacritics` | `False` | drop Arabic harakat (U+064B to U+065F, U+0670) |
| `char_map` | `{}` | single-character replacements, e.g. `{"ك": "ک"}` maps Arabic kaf to Sindhi kaf |

```python
opts = CleaningOptions(strip_diacritics=True)
clean_text("سَٺو", opts)  # 'سٺو'
clean_text("ڪتاب‌ڪ", CleaningOptions(remove_zwnj=True))  # 'ڪتابڪ'
clean_text("ك", CleaningOptions(char_map={"ك": "ک"}))  # 'ک'
```

`char_map` keys must be exactly one character; unknown option names raise `ValueError`.

## Where the options come from

| Stage | Source |
| --- | --- |
| data preparation and training | `data.cleaning` in `configs/base.yaml` |
| prediction | `preprocessing.json` in the model folder; if missing, the defaults above |

`preprocessing.json` is how a published model carries its cleaning settings with it:

```json
{
  "cleaning": {
    "char_map": {},
    "remove_invisible": true,
    "remove_tatweel": true,
    "remove_zwnj": false,
    "strip_diacritics": false,
    "unicode_form": "NFKC"
  },
  "source": "configs/base.yaml data.cleaning"
}
```

Write it with `sindhi-nlp finalize-model --model-dir <folder>` (it reads `configs/base.yaml`), or from Python:

```python
from sindhi_nlp.data.clean_data import CleaningOptions
from sindhi_nlp.inference.preprocessing import write_preprocessing_config

write_preprocessing_config("artifacts/models/my_model", CleaningOptions(strip_diacritics=True), source="my run")
```

!!! danger "Never change cleaning after training"
    If you train with `strip_diacritics: true` and then predict without `preprocessing.json`, the model sees text it was never trained on and accuracy drops silently. Always run `finalize-model` after training and publish `preprocessing.json` with the model.

## Validation at prediction time

`TextPreprocessor.prepare(text)` is what `SentimentPredictor` calls on every input:

```python
from sindhi_nlp.inference.preprocessing import TextPreprocessor

pre = TextPreprocessor()  # defaults; or TextPreprocessor.from_model_dir("path/to/model")
pre.prepare("  اڄ  سٺو آهي ")  # 'اڄ سٺو آهي'
pre.prepare("   ")  # InvalidInputError: Text is empty after cleaning
pre.prepare(123)  # InvalidInputError: Text must be a string, got int
```

Texts longer than `max_chars` (default 10,000 characters) are rejected before cleaning. Texts longer than the model's **token** limit are truncated by the tokenizer instead (128 tokens for the published model); the longest sentence in the training data, 54 words, fits comfortably.

## Tips

- Audit your own text before trusting any model: count letter variants such as Arabic vs Sindhi kaf and yeh, and decide on a `char_map` *before* training, using validation data.
- Copy-pasted text from web pages often contains invisible characters. The defaults handle them; you do not need to clean text yourself before calling the predictor.
- Keep the original text. `Prediction.text` is the text you passed in, not the cleaned version.
