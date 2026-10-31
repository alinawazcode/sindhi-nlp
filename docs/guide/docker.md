# Docker

The repository's `Dockerfile` serves the [HTTP API](../api.md) on port 8000. The model weights are **not** baked into the image: you mount a model folder at run time, or point the config at the published Hugging Face model.

## With docker compose

From the repository root, with a trained model in `artifacts/models/`:

```bash
docker compose up --build
```

Then open <http://127.0.0.1:8000/docs>. `docker-compose.yml` mounts `./artifacts/models` and `./configs/inference.yaml` read-only, so the `model_dir` in `configs/inference.yaml` must exist inside that folder.

## With docker run

```bash
docker build -t sindhi-nlp-api .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/artifacts/models:/app/artifacts/models:ro" \
  sindhi-nlp-api
```

On PowerShell replace `$(pwd)` with `${PWD}`. The image contains `configs/`, so `model_dir` there must match the mounted folder.

## What the image contains

- `python:3.12-slim`, CPU-only PyTorch (much smaller than the default CUDA build), and `sindhi-nlp[api,transformers]`.
- A non-root user (`appuser`, uid 1000).
- A health check that calls `GET /health` every 30 seconds, with a 90-second start period for model loading.

## Notes

- Run **one worker**. Every extra worker loads its own copy of the model.
- The service has no authentication. Put it behind a reverse proxy before exposing it.
- To serve the published model, set `model_dir: alinawazmahar/sindhi-sentiment-xlmr` in the mounted `inference.yaml`. The container then needs internet access on first start and a writable cache; mount a volume at the Hugging Face cache (`HF_HOME`) to keep it between runs.
- These Docker files were not built or run while writing this documentation. Build them in your own environment before depending on them.
