# Python API reference

This page lists the public API. Each item also has a docstring (`help(SentimentPredictor.predict)`), and the package ships type information (`py.typed`), so editors and `mypy` understand it.

## Top level (`sindhi_nlp`)

| Name | Kind | Description |
| --- | --- | --- |
| `__version__` | `str` | installed version |
| `SentimentPredictor` | class | load a model and classify text |
| `Prediction` | dataclass | one result |
| `ModelInfo` | dataclass | details of a loaded model |
| `InvalidInputError` | exception | unusable text |
| `ModelNotFoundError` | exception | model folder or files missing |
| `ModelLoadError` | exception | model files unusable |

Names are loaded lazily, so `import sindhi_nlp` does not import PyTorch.

## `SentimentPredictor`

### `from_pretrained(model_dir, device="auto", batch_size=32, max_length=None, confidence_threshold=None, max_chars=None, revision=None, cache_dir=None)`

Class method. Returns a predictor from a local folder or a Hugging Face name. Raises `ModelNotFoundError`, `ModelLoadError`, `ValueError`. Details: [Python SDK](../guide/python-sdk.md#loading-a-model).

### `from_config(config_path="configs/inference.yaml", project_root=".")`

Class method. Builds a predictor from a YAML file. Raises `ConfigError` for a missing file or invalid `device`.

### `predict(text) -> Prediction`

Classify one text. Raises `InvalidInputError`.

### `predict_batch(texts, batch_size=None, errors="raise") -> list[Prediction | None]`

Classify many texts in order. `errors="raise"` raises `InvalidInputError("Item <i>: ...")`; `errors="none"` returns `None` for bad items. Raises `TypeError` for a single string and `ValueError` for a bad `errors` or `batch_size`.

### Properties

| Property | Type | Description |
| --- | --- | --- |
| `info` | `ModelInfo` | model details |
| `labels` | `tuple[str, ...]` | label order |

## `Prediction`

Frozen dataclass.

| Field | Type | Description |
| --- | --- | --- |
| `text` | `str` | the text as passed in |
| `label` | `str` | predicted label |
| `confidence` | `float` | probability or score of `label` |
| `probabilities` | `dict[str, float]` | per label, sums to 1 |
| `is_confident` | `bool` | `True` unless below `confidence_threshold` |

Method: `to_dict() -> dict`.

## `ModelInfo`

Frozen dataclass.

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Hub name or folder name |
| `backend` | `str` | `"sklearn"` or `"transformers"` |
| `model_dir` | `str` | local folder in use |
| `labels` | `tuple[str, ...]` | label order |
| `confidence_type` | `str` | `"probability"` or `"score"` |
| `device` | `str` | `"cpu"` or `"cuda"` |
| `max_length` | `int \| None` | token limit (transformers only) |
| `preprocessing_source` | `str` | where cleaning options came from |
| `seed` | `int \| None` | training seed, if recorded |
| `base_model` | `str \| None` | e.g. `xlm-roberta-base` |

Method: `to_dict() -> dict`.

## Preprocessing (`sindhi_nlp.inference.preprocessing`, `sindhi_nlp.data.clean_data`)

| Name | Description |
| --- | --- |
| `clean_text(text, options=DEFAULT_OPTIONS) -> str` | normalize one text; non-strings become `""` |
| `clean_series(series, options)` | apply to a pandas Series |
| `CleaningOptions(...)` | cleaning switches; `from_config(dict)`, `to_dict()` |
| `TextPreprocessor(options=None, max_chars=10000, source="defaults")` | `prepare(text) -> str`, `from_model_dir(path)` |
| `write_preprocessing_config(model_dir, options, source="")` | write `preprocessing.json` |

See [Text preprocessing](../guide/preprocessing.md).

## Evaluation (`sindhi_nlp.evaluation`)

The `evaluate`, `error_analysis` and `plots` modules were added in 0.1.2; see [Which version has what](../installation.md#which-version-has-what).

| Name | Description |
| --- | --- |
| `evaluate.evaluate_predictor(predictor, texts, y_true)` | returns `(metrics, results_frame)`; skips unclassifiable texts |
| `evaluate.evaluate_predictions(y_true, y_pred, labels)` | metrics dictionary |
| `evaluate.format_report(metrics, title="")` | the text report |
| `error_analysis.build_results_frame(texts, y_true, y_pred, confidences=None)` | table with `text, true, predicted, confidence, correct` |
| `error_analysis.misclassified(frame, limit=None)` | mistakes, most confident first |
| `error_analysis.confusion_pairs(frame)` | counts of `(true, predicted)` mistakes |
| `plots.plot_confusion_matrix(matrix, labels, path=None, normalize=False, title, dpi)` | needs `sindhi-nlp[plots]` |
| `plots.plot_model_comparison(scores, path, metric, title, dpi)` | bar chart of one metric per model |
| `plots.plot_per_label_f1(per_label, path, title, dpi)` | per-class F1 bars; `per_label` as in the metrics dictionary |

The plot functions return a matplotlib `Figure` and write a PNG when `path` is given. See [Evaluation tools](evaluation.md).

## Configuration and utilities (`sindhi_nlp.utils`)

| Name | Description |
| --- | --- |
| `config.load_yaml`, `load_config(path, required=())`, `get_nested(config, "a.b")`, `require_keys`, `merge_configs`, `ConfigError` | YAML config helpers |
| `seed.set_seed(seed)` | seed Python, NumPy and (if installed) PyTorch |
| `logging.setup_logging(level)`, `get_logger(name)` | consistent logging on stderr |

## Stability

`SentimentPredictor`, `Prediction`, `ModelInfo`, the three exceptions and the `sindhi-nlp` command are intended to stay stable within 0.x releases, with changes announced in the [changelog](../changelog.md). Everything else may change in a minor release.
