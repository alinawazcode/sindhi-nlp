# Python SDK

Everything lives behind one class, `SentimentPredictor`, plus three result and error types. Importing `sindhi_nlp` is cheap: PyTorch and scikit-learn load only when you create a predictor.

```python
from sindhi_nlp import (
    SentimentPredictor,  # load a model and classify text
    Prediction,  # one result
    ModelInfo,  # details of a loaded model
    InvalidInputError,  # bad text
    ModelNotFoundError,  # model folder / files missing
    ModelLoadError,  # model files present but unusable
)
```

## Loading a model

### `SentimentPredictor.from_pretrained`

```python
SentimentPredictor.from_pretrained(
    model_dir,  # local folder, or Hugging Face name "owner/model"
    device="auto",  # "auto" | "cpu" | "cuda"
    batch_size=32,  # texts per forward pass
    max_length=None,  # token limit; default: the value saved with the model (128 for the published one)
    confidence_threshold=None,  # 0..1; flags low-confidence results
    max_chars=None,  # reject longer texts (default 10,000 characters)
    revision=None,  # Hub only: branch, tag or commit hash
    cache_dir=None,  # Hub only: where to cache the download
)
```

| Argument | Notes |
| --- | --- |
| `model_dir` | A `str` or `Path`. A local folder always wins. A `str` shaped like `owner/name` that is not a local folder is downloaded from the Hugging Face Hub. A `Path` is never treated as a Hub name, so a typo in a local path cannot trigger a download. |
| `device` | `"auto"` uses CUDA if available, otherwise CPU. Ignored by the scikit-learn baseline, which always runs on the CPU. |
| `batch_size` | Larger is faster on a GPU and uses more memory. |
| `max_length` | Texts longer than this many tokens are truncated, not rejected. |
| `confidence_threshold` | Only affects `Prediction.is_confident`. It never changes the label. |
| `revision` | Pin an exact model version for reproducible deployments, e.g. a commit hash. |

The model type is detected from the files in the folder: `model.joblib` means the TF-IDF baseline; `model.safetensors` (or `pytorch_model.bin`) means a transformers model.

```python
# published model, downloaded once and cached
predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr")

# a pinned version
predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr", revision="main")

# a local folder, CPU only, flag results below 0.9
predictor = SentimentPredictor.from_pretrained("artifacts/models/xlmr_mixed", device="cpu", confidence_threshold=0.9)

# the lightweight baseline
baseline = SentimentPredictor.from_pretrained("artifacts/models/baseline_natural")
```

### `SentimentPredictor.from_config`

Builds a predictor from a YAML file (default `configs/inference.yaml`, relative to `project_root`):

```python
predictor = SentimentPredictor.from_config("configs/inference.yaml", project_root=".")
```

`model_dir` is required. `device`, `batch_size`, `max_length`, `confidence_threshold`, `max_chars` and `revision` are optional. See [Configuration files](../reference/configuration.md#inference-yaml).

## Predicting

### One text: `predict`

```python
result = predictor.predict("اڄ جو ڏينهن تمام سٺو آهي")
result.label  # "positive"
result.confidence  # 0.99...
result.probabilities  # {"negative": ..., "neutral": ..., "positive": ...}
result.is_confident  # True
result.to_dict()  # plain dict, ready for json.dumps(..., ensure_ascii=False)
```

`Prediction` is a frozen dataclass: you cannot change its fields after creation, and two predictions compare equal when all fields match. It holds a dictionary, so it is not hashable and cannot be a set member or dict key.

### Many texts: `predict_batch`

```python
results = predictor.predict_batch(texts)  # list[Prediction]
results = predictor.predict_batch(texts, batch_size=64)  # override the batch size for this call
```

- Any iterable of strings works (list, tuple, generator, pandas Series via `.tolist()`).
- The output has one entry per input, in the same order.
- A single string is rejected with `TypeError`, because iterating a string would classify each character. Use `predict` for one text.

### Handling bad input

A text is invalid if it is not a `str`, is longer than `max_chars`, or is empty after cleaning (for example only spaces or invisible characters).

```python
from sindhi_nlp import InvalidInputError

try:
    predictor.predict_batch(["اڄ سٺو آهي", "   ", "ٻيو جملو"])
except InvalidInputError as exc:
    print(exc)  # Item 1: Text is empty after cleaning
```

With the default `errors="raise"`, the first bad item stops the call and the message names its position. To classify the good items and get `None` for the bad ones:

```python
results = predictor.predict_batch(["اڄ سٺو آهي", "   ", "ٻيو جملو"], errors="none")
for text, result in zip(texts, results):
    if result is None:
        print("skipped:", repr(text))
    else:
        print(result.label)
```

!!! tip "Typing"
    `predict_batch` returns `list[Prediction | None]` because of `errors="none"`. With the default you can safely treat items as `Prediction`; type checkers will ask you to narrow them.

### Filtering by confidence

```python
predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr", confidence_threshold=0.8)
reviewed = [r for r in predictor.predict_batch(texts) if not r.is_confident]
```

Because XLM-R confidence is not calibrated (see [Confidence scores](../concepts/confidence.md)), use a threshold to *route items to a person*, not as a guarantee of accuracy.

## Inspecting a model

```python
info = predictor.info  # ModelInfo
info.name  # "alinawazmahar/sindhi-sentiment-xlmr"
info.backend  # "transformers" or "sklearn"
info.labels  # ("negative", "neutral", "positive")
info.confidence_type  # "probability" (transformers) or "score" (sklearn baseline)
info.device, info.max_length, info.seed, info.base_model
info.preprocessing_source  # "preprocessing.json" or "defaults (no preprocessing.json found)"
predictor.labels  # same as info.labels
```

`ModelInfo.to_dict()` gives a JSON-ready dictionary (labels become a list).

## Working with pandas

```python
import pandas as pd

df = pd.read_csv("reviews.csv", encoding="utf-8-sig")
results = predictor.predict_batch(df["Text"].astype(str).tolist(), errors="none")

df["label"] = [r.label if r else None for r in results]
df["confidence"] = [r.confidence if r else None for r in results]
df.to_csv("reviews_scored.csv", index=False, encoding="utf-8-sig")
```

Use `utf-8-sig` so Excel opens Sindhi text correctly.

## Thread safety and performance

- A predictor holds a model in memory and is **not** guaranteed safe for concurrent calls from several threads. Use one predictor per process and serialize calls with a lock (the HTTP API does this for you).
- Load the model **once** and reuse it. Loading XLM-R takes seconds; predicting one sentence takes milliseconds to a fraction of a second.
- For throughput on a GPU, raise `batch_size`. On a CPU, the baseline is far faster than XLM-R; see [Choosing a model](choosing-a-model.md).

## Errors

| Exception | Raised when | Base class |
| --- | --- | --- |
| `InvalidInputError` | text is not a string, too long, or empty after cleaning | `ValueError` |
| `ModelNotFoundError` | the folder does not exist, has no model files, or a Hub download fails | `FileNotFoundError` |
| `ModelLoadError` | files exist but cannot be loaded, or a needed library is missing | `RuntimeError` |
| `ValueError` | invalid arguments such as `batch_size=0` or a threshold outside 0..1 | built-in |
| `TypeError` | a single string was passed to `predict_batch` | built-in |

See [Errors](../reference/errors.md) for messages and fixes.

## Text cleaning

Every text is cleaned with the same function used in training before it reaches the model. See [Text preprocessing](preprocessing.md).

## Complete example

```python
from sindhi_nlp import InvalidInputError, SentimentPredictor


def classify(texts: list[str]) -> list[dict]:
    predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr", confidence_threshold=0.8)
    out = []
    for text, result in zip(texts, predictor.predict_batch(texts, errors="none")):
        if result is None:
            out.append({"text": text, "label": None, "needs_review": True})
        else:
            out.append({"text": text, "label": result.label, "needs_review": not result.is_confident})
    return out
```
