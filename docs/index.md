# sindhi-nlp

Sentiment classification for Sindhi text. Give it a Sindhi sentence; it returns `negative`, `neutral` or `positive`, with class probabilities.

```python
from sindhi_nlp import SentimentPredictor

predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr")
result = predictor.predict("اڄ جو ڏينهن تمام سٺو آهي")
print(result.label, result.confidence)
```

The project ships everything around the model, not just the model:

| You want to... | Use | Read |
| --- | --- | --- |
| classify text from Python | `SentimentPredictor` | [Python SDK](guide/python-sdk.md) |
| classify a file from a terminal | `sindhi-nlp` command | [Command line](guide/cli.md) |
| call it from another service | FastAPI app | [HTTP API](api.md) |
| measure it on your own labeled data | `python -m sindhi_nlp.evaluation.evaluate` | [Evaluate on your own data](tutorials/evaluate-own-data.md) |
| train or fine-tune your own model | training pipeline | [Train on your own data](tutorials/train-own-data.md) |
| understand how far to trust it | results and limits | [Results](results.md), [Model card](model_card.md) |

## What is in the box

- **Two models.** A fine-tuned XLM-RoBERTa base (`alinawazmahar/sindhi-sentiment-xlmr`, about 1.1 GB) and a TF-IDF + LinearSVC baseline you can train in under a second on a CPU.
- **A Python SDK** with typed results, batch prediction, per-item error handling and the same text cleaning used in training.
- **A command line** for single texts, CSV batches and model details.
- **An HTTP API** with a consistent error format, request validation and a batch endpoint.
- **A reproducible pipeline**: seeded, leakage-aware data splits, training, evaluation, paired significance testing, and a one-time test-set rule.
- **Evaluation tools** for scoring any saved model on your own labeled CSV, listing its most confident mistakes and drawing confusion matrices.

## Read this before trusting the numbers

!!! warning "Two datasets, one honest number"
    The project started with a 150,000-sentence synthetic dataset on which every model scores 0.9998. That says nothing about real Sindhi: the same baseline scores 0.60 on human-written sentences. All headline results come from a separate human-labeled set of 4,420 sentences. The best model reaches **0.891 test macro-F1**, and its lead over the TF-IDF baseline is **not statistically significant** (p = 0.116).

See [Results](results.md) for the full numbers and [Roadmap and known gaps](roadmap.md) for what is not done yet.

## Status

Version 0.1.x, alpha. The API of `SentimentPredictor` and the CLI is intended to stay stable; the internal modules (`sindhi_nlp.data`, `sindhi_nlp.training`) may change between minor versions. See the [changelog](changelog.md).
