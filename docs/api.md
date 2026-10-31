# Sindhi sentiment API

A small HTTP service that classifies Sindhi text as `negative`, `neutral`, or `positive`. It is built on the same `SentimentPredictor` as the Python SDK, so the text is cleaned exactly as it was for training.

## Install and run

```powershell
pip install fastapi uvicorn
$env:PYTHONPATH="src"
python -m uvicorn sindhi_nlp.api.main:app --host 127.0.0.1 --port 8000
```

Run it from the project root. The model named in `configs/inference.yaml` (`model_dir`) is loaded once, when the server starts, which takes a few seconds for XLM-R. Interactive documentation is at <http://127.0.0.1:8000/docs>.

Settings live in `configs/inference.yaml`:

| Setting | Meaning |
| --- | --- |
| `model_dir` | Folder of the model to serve, e.g. `artifacts/models/xlmr_mixed` or `artifacts/models/baseline_natural` |
| `device` | `auto`, `cpu`, or `cuda` |
| `confidence_threshold` | Optional. Predictions below it come back with `is_confident: false` |
| `api.max_batch_size` | Most texts accepted by `/predict/batch` (default 64) |
| `api.cors_origins` | Browser origins allowed to call the API (default none) |

The environment variables `SINDHI_NLP_CONFIG` and `SINDHI_NLP_ROOT` point the server at a different config file or project folder.

Run a single worker. Each extra worker process loads its own copy of the model.

## Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "xlmr_mixed",
  "backend": "transformers",
  "package_version": "0.1.1",
  "detail": null
}
```

This is real output from a local run on a Windows CPU.

Returns **503** with `"status": "unavailable"` if the model failed to load. The server stays up so you can see why; the details are in the server log.

### `GET /model`

Labels, backend, device, token limit, and which preprocessing settings are in use. Real output:

```json
{
  "name": "xlmr_mixed",
  "backend": "transformers",
  "labels": ["negative", "neutral", "positive"],
  "confidence_type": "probability",
  "device": "cpu",
  "max_length": 128,
  "preprocessing_source": "preprocessing.json",
  "seed": 42,
  "base_model": "xlm-roberta-base"
}
```

### `POST /predict`

Request:

```json
{ "text": "اڄ جو ڏينهن تمام سٺو آهي" }
```

Response (the numbers are illustrative):

```json
{
  "text": "اڄ جو ڏينهن تمام سٺو آهي",
  "label": "positive",
  "confidence": 0.99,
  "probabilities": { "negative": 0.004, "neutral": 0.006, "positive": 0.99 },
  "is_confident": true,
  "model": "xlmr_mixed"
}
```

- `text` is returned exactly as you sent it. Cleaning (Unicode normalization, invisible characters, extra spaces) is applied internally before classification.
- Texts must be 1 to 10,000 characters, and not empty after cleaning.

### `POST /predict/batch`

Request:

```json
{ "texts": ["اڄ جو ڏينهن تمام سٺو آهي", "مون کي ڏاڍو ڏک ٿيو"] }
```

Response:

```json
{
  "model": "xlmr_mixed",
  "count": 2,
  "results": [ { "text": "...", "label": "positive", "confidence": 0.99, "probabilities": {}, "is_confident": true },
               { "text": "...", "label": "negative", "confidence": 0.99, "probabilities": {}, "is_confident": true } ]
}
```

Results keep the order of the request. If any text is invalid, the whole request is rejected with an error that names the position (`Item 1: ...`); nothing is partially classified.

## About `confidence`

- For the XLM-R model it is a softmax probability. It tends to be **overconfident**: wrong predictions can also come back near 1.0, so do not treat it as a calibrated probability or use it alone to filter results.
- For the TF-IDF baseline it is a softmax of classifier margins: a ranking score, not a probability. `GET /model` shows which kind applies (`confidence_type`).

## Errors

Every error uses the same shape. The submitted text is never echoed back or logged.

```json
{ "error": { "code": "invalid_input", "message": "Text is empty after cleaning" } }
```

| Status | `code` | When |
| --- | --- | --- |
| 422 | `validation_error` | Missing field, wrong type, empty or over-long text, malformed JSON. `details` lists the fields. |
| 422 | `invalid_input` | Text is empty after cleaning (for example only spaces) |
| 422 | `batch_too_large` | More texts than `api.max_batch_size` |
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown path or wrong HTTP method |
| 503 | `model_unavailable` | The model is not loaded |
| 500 | `internal_error` | Unexpected failure; details are only in the server log |

## Calling the API

**Python** (recommended; handles Sindhi text correctly):

```python
import requests

reply = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "اڄ جو ڏينهن تمام سٺو آهي"},
)
print(reply.json()["label"])
```

**curl.exe on Windows.** Put the JSON in a UTF-8 file, because Sindhi text typed directly into a PowerShell command can be mangled by the console encoding:

```powershell
'{"text": "اڄ جو ڏينهن تمام سٺو آهي"}' | Out-File -Encoding utf8NoBOM body.json
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@body.json"
```

`utf8NoBOM` needs PowerShell 7. In Windows PowerShell 5.1, create `body.json` in an editor that saves UTF-8 without a BOM.

## Notes

- Inference runs one request at a time (a lock protects the model), so throughput is bounded by model speed. XLM-R on CPU handles a few sentences per second; use a GPU or the baseline model for more.
- Inputs longer than the model's token limit are truncated (128 tokens for the current XLM-R model).
- There is no authentication or rate limiting. Put the service behind a reverse proxy or API gateway before exposing it to the internet.