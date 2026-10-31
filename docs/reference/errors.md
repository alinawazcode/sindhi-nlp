# Errors

What each error means and what to do about it.

## Python exceptions

| Exception | Message example | Cause and fix |
| --- | --- | --- |
| `InvalidInputError` | `Text is empty after cleaning` | the text is blank or only invisible characters; skip or fix the row |
| `InvalidInputError` | `Text must be a string, got float` | a pandas `NaN` or number; use `.astype(str)` or `errors="none"` |
| `InvalidInputError` | `Text has 12,000 characters; the limit is 10,000` | raise `max_chars`, or split the text into sentences |
| `InvalidInputError` | `Item 3: ...` | from `predict_batch`; the bad item's position is 3 (zero-based) |
| `ModelNotFoundError` | `Model folder not found: ...` | wrong path; folders resolve against the current directory |
| `ModelNotFoundError` | `No model files in ...: expected model.joblib or ...` | the folder is not a model folder |
| `ModelNotFoundError` | `'owner/name' is not a local folder and could not be downloaded` | typo, no internet, or no access to the repository |
| `ModelLoadError` | `Loading a model by name needs: pip install huggingface_hub` | install `sindhi-nlp[hub]` or `[transformers]` |
| `ModelLoadError` | `Invalid .../preprocessing.json: ...` | the file is malformed; regenerate with `sindhi-nlp finalize-model` |
| `ModelLoadError` | `The model returned scores of shape ...` | the model does not match its label list |
| `ValueError` | `batch_size must be at least 1`, `confidence_threshold must be between 0 and 1` | invalid argument |
| `TypeError` | `predict_batch expects a list of texts` | you passed one string; use `predict` |
| `ConfigError` | `Config file not found`, `Missing required config key(s): model_dir` | fix the YAML file |

## Command line

Handled errors print `Error: <message>` on stderr and exit with code 1. `argparse` usage errors exit with 2. A `batch` run never fails because of a bad row; the row is left blank with `error = invalid input`.

## HTTP API

See [HTTP API: Errors](../api.md#errors) for status codes and the JSON error shape.

## Data and evaluation tools

| Message | Cause and fix |
| --- | --- |
| `The test split is a one-time final evaluation. Pass --final to confirm.` | add `--final` once you are really finished choosing models |
| `<model>_external_test.json already exists: the test split was already scored for this model` | by design; use `--overwrite` only when you are certain, knowing the test set has been seen |
| `Column 'Text' not found in file.csv; columns are [...]` | pass `--text-column` / `--label-column` |
| `Gold labels [...] are not labels of this model [...]` | your labels must be `negative`, `neutral`, `positive` |
| `None of the texts could be classified` | all rows were empty or invalid |
| `out of data/raw` | the human-labeled file must not be placed in `data/raw/` |
