"""Call the HTTP API from Python.

First start the server in another terminal (see the "Serve the API" tutorial):

    pip install "sindhi-nlp[api,transformers]" requests
    # create configs/inference.yaml containing:  model_dir: alinawazmahar/sindhi-sentiment-xlmr
    uvicorn sindhi_nlp.api.main:app --host 127.0.0.1 --port 8000

Then run:  python examples/04_serve_and_call_api.py
"""

import requests

BASE = "http://127.0.0.1:8000"

health = requests.get(f"{BASE}/health", timeout=10)
print("health:", health.status_code, health.json())
if health.status_code != 200:
    raise SystemExit("The server is up but the model is not loaded; check the server log.")

one = requests.post(f"{BASE}/predict", json={"text": "اڄ جو ڏينهن تمام سٺو آهي"}, timeout=30)
one.raise_for_status()
print(one.json()["label"], one.json()["confidence"])

many = requests.post(
    f"{BASE}/predict/batch",
    json={"texts": ["اڄ جو ڏينهن تمام سٺو آهي", "مون کي ڏاڍو ڏک ٿيو"]},
    timeout=60,
)
many.raise_for_status()
for item in many.json()["results"]:
    print(item["label"], item["confidence"], item["text"])

# Errors have one shape: {"error": {"code": ..., "message": ...}}
bad = requests.post(f"{BASE}/predict", json={"text": "   "}, timeout=30)
print(bad.status_code, bad.json()["error"]["code"])  # 422 invalid_input
