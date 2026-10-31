# Serve the API

Goal: run the HTTP service locally, call it from Python and from the command line, and know what to do before exposing it.

The endpoint reference is on the [HTTP API](../api.md) page; this tutorial is the walk-through.

## 1. Install and start

```bash
pip install "sindhi-nlp[api,transformers]"
```

The server loads the model named in `configs/inference.yaml` (`model_dir`) relative to the folder you start it from. If you installed from PyPI there is no such file; create one:

```yaml
# configs/inference.yaml
model_dir: alinawazmahar/sindhi-sentiment-xlmr
device: auto
confidence_threshold: null
api:
  max_batch_size: 64
  cors_origins: []
```

Start it:

```bash
uvicorn sindhi_nlp.api.main:app --host 127.0.0.1 --port 8000
```

Startup takes a few seconds for XLM-R (and downloads the model on first use). Run a **single** worker; each extra worker loads another copy.

## 2. Check that it is healthy

```bash
curl http://127.0.0.1:8000/health
```

`"model_loaded": true` means it is ready. If loading failed the server stays up, `/health` returns 503 with the reason, and the server log has details.

Interactive documentation, generated from the code, is at <http://127.0.0.1:8000/docs>.

## 3. Call it from Python

```python
import requests

BASE = "http://127.0.0.1:8000"

one = requests.post(f"{BASE}/predict", json={"text": "اڄ جو ڏينهن تمام سٺو آهي"}, timeout=30)
print(one.json()["label"])

many = requests.post(
    f"{BASE}/predict/batch",
    json={"texts": ["اڄ جو ڏينهن تمام سٺو آهي", "مون کي ڏاڍو ڏک ٿيو"]},
    timeout=60,
)
for item in many.json()["results"]:
    print(item["label"], item["confidence"])
```

## 4. Handle errors

Every error has the same shape, so one handler is enough:

```python
reply = requests.post(f"{BASE}/predict", json={"text": "   "}, timeout=30)
if reply.status_code != 200:
    error = reply.json()["error"]
    print(error["code"], error["message"])  # invalid_input  Text is empty after cleaning
```

The codes are listed in [HTTP API: Errors](../api.md#errors). A batch with any invalid text is rejected as a whole and names the position (`Item 1: ...`).

## 5. Configure it

| Setting | Where | Effect |
| --- | --- | --- |
| `model_dir`, `device`, `confidence_threshold`, `max_chars` | `configs/inference.yaml` | which model and how it behaves |
| `api.max_batch_size` | same file | the largest accepted batch |
| `api.cors_origins` | same file | browser origins allowed to call the API |
| `SINDHI_NLP_CONFIG`, `SINDHI_NLP_ROOT` | environment | use another config file or project folder |

PowerShell: `$env:SINDHI_NLP_CONFIG="configs/prod.yaml"` before starting uvicorn.

## 6. Before you expose it

The service has **no authentication and no rate limiting**, and runs inference one request at a time. For anything beyond your own machine:

- put it behind a reverse proxy or gateway that handles TLS, authentication and rate limits;
- keep `--host 127.0.0.1` unless the proxy is on another machine;
- set `api.cors_origins` only to the origins that need it;
- monitor `/health`;
- remember throughput is bounded by the model: XLM-R on a CPU handles a few sentences per second. Use the baseline or a GPU for more.

See also [Docker](../guide/docker.md).
