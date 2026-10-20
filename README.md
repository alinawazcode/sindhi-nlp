# sindhi-nlp

[![tests](https://github.com/alinawazcode/sindhi-nlp/actions/workflows/tests.yml/badge.svg)](https://github.com/alinawazcode/sindhi-nlp/actions/workflows/tests.yml)

Sentiment classification for Sindhi text. Given a Sindhi sentence, it returns `negative`, `neutral`, or `positive` with class probabilities.

The repository contains a reproducible training pipeline, a fine-tuned XLM-RoBERTa model, a lightweight TF-IDF baseline, a Python SDK, a command-line tool, and an HTTP API.

> **Read this before trusting the numbers.** The project began with a 150,000-sentence synthetic dataset on which every model scores 0.9998. That figure says nothing about real Sindhi: the same baseline scores 0.60 on human-written sentences. All headline results below come from a separate human-labeled dataset. See [Data](#data) and [Limitations](#limitations).

## Results

Evaluated on 4,420 human-labeled Sindhi sentences (3,094 train / 442 validation / 884 test). The test split was scored once per model, after all choices were fixed.

| Model | Trained on | Validation macro-F1 | Test macro-F1 | Test accuracy |
| --- | --- | ---: | ---: | ---: |
| TF-IDF + LinearSVC | synthetic data only | 0.6002 | not run | not run |
| TF-IDF + LinearSVC (`baseline_natural`) | human-labeled sentences | 0.8665 | 0.8695 | 0.8688 |
| XLM-RoBERTa base | human-labeled sentences | 0.8663 | not run | not run |
| **XLM-RoBERTa base (`xlmr_mixed`)** | up to 30,000 synthetic + human-labeled (repeated 5x) | **0.8872** | **0.8906** | **0.8903** |

XLM-R scores higher, but **the difference from the baseline is not statistically significant** on this test set (exact McNemar test, p = 0.116; the baseline alone was wrong on 75 sentences, XLM-R alone on 56). Each accuracy carries a 95% interval of roughly ±2 points (baseline 0.845 to 0.889, XLM-R 0.868 to 0.909). The baseline trains in under a second on a CPU, so it is a reasonable choice when resources are limited.

Per-class scores, confusion matrices, and the paired comparison are written to `artifacts/metrics/` when you run the evaluation commands below.

## Install

Requires Python 3.11 or newer.

```bash
pip install "sindhi-nlp[transformers]"    # SDK, CLI, and the XLM-R model
pip install "sindhi-nlp[api]"             # add the HTTP API
```

The first time you use the published model it downloads about 1.1 GB, which takes a few minutes on a slow connection. It is cached afterwards.

To work on the code, clone the repository and install it in editable mode with the test tools:

```bash
git clone https://github.com/alinawazcode/sindhi-nlp.git
cd sindhi-nlp
pip install -e ".[transformers,api,dev]"
```

Model weights are not stored in Git. The trained XLM-R model is published on Hugging Face as [`alinawazmahar/sindhi-sentiment-xlmr`](https://huggingface.co/alinawazmahar/sindhi-sentiment-xlmr) and is downloaded and cached automatically the first time you use that name. You can also point at a local model folder (for example `artifacts/models/xlmr_mixed`) or train your own with the steps in [Reproduce the results](#reproduce-the-results).

## Use it

**Python**

```python
from sindhi_nlp import SentimentPredictor

# a Hugging Face name (downloaded once, then cached) or a local folder
predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr")

result = predictor.predict("اڄ جو ڏينهن تمام سٺو آهي")
print(result.label)  # positive
print(result.probabilities)  # {'negative': ..., 'neutral': ..., 'positive': ...}

results = predictor.predict_batch(["...", "..."])  # keeps the input order
```

Input is cleaned the same way as during training. Empty or non-text input raises `InvalidInputError`, and inputs beyond the model's token limit are truncated.

**Command line**

```bash
sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي" --model-dir alinawazmahar/sindhi-sentiment-xlmr
sindhi-nlp batch --input texts.csv --output results.csv --model-dir artifacts/models/xlmr_mixed
sindhi-nlp info --model-dir artifacts/models/xlmr_mixed
```

**HTTP API**

```bash
uvicorn sindhi_nlp.api.main:app --host 127.0.0.1 --port 8000
```

The model named in `configs/inference.yaml` is loaded at startup. Open <http://127.0.0.1:8000/docs> for interactive documentation, or see [docs/api.md](https://github.com/alinawazcode/sindhi-nlp/blob/main/docs/api.md) for endpoints, error formats, and examples.

> **About `confidence`.** XLM-R's probabilities are overconfident: wrong predictions can also come back near 1.0. Treat the value as a ranking score, not a calibrated probability.

## Data

Two datasets are used. Details are in [docs/dataset.md](https://github.com/alinawazcode/sindhi-nlp/blob/main/docs/dataset.md).

- **Human-labeled dataset (4,420 sentences).** Collected from Awaz and other articles and newspapers and labeled by one annotator. Used for all reported results. It is published separately on Kaggle: <https://www.kaggle.com/datasets/alinawaz06/sindhi-sentiment-analysis-dataset>. The sentences come from published sources, so check the dataset page for its terms before reusing it.
- **Synthetic dataset (150,000 sentences).** Machine-generated from sentence templates with a vocabulary of about 2,400 words. It is useful for testing the pipeline but is not representative of real Sindhi. It is not included in this repository.

## Reproduce the results

With the data in place (`data/raw/` for the synthetic CSV, `data/external/natural_sindhi.csv` for the human-labeled one):

```bash
# 1. Build the splits
python -m sindhi_nlp.data.split_data --config configs/base.yaml
python -m sindhi_nlp.data.prepare_external

# 2. Baselines (seconds on a CPU)
python -m sindhi_nlp.training.baseline
python -m sindhi_nlp.training.baseline --data-dir data/external/processed --model-name baseline_natural

# 3. XLM-RoBERTa (needs a GPU; a free Kaggle T4 works)
python -m sindhi_nlp.training.train --smoke      # quick check of the whole pipeline
python -m sindhi_nlp.training.train

# 4. Evaluate. Validation can be repeated; the test split is a one-time evaluation.
python -m sindhi_nlp.evaluation.evaluate_external --model-dir artifacts/models/xlmr_mixed --split validation
python -m sindhi_nlp.evaluation.evaluate_external --model-dir artifacts/models/xlmr_mixed --split test --final
python -m sindhi_nlp.evaluation.compare_models --a baseline_natural --b xlmr_mixed --split test
```

Every run is seeded (42) and writes its configuration, metrics, and data hashes next to the model, so results can be checked later. See [docs/architecture.md](https://github.com/alinawazcode/sindhi-nlp/blob/main/docs/architecture.md) for how the pieces fit together.

## Project layout

```
configs/      YAML settings for data, training, evaluation, and inference
src/sindhi_nlp/
  data/        loading, cleaning, validation, splitting
  training/    TF-IDF baseline, XLM-R fine-tuning, metrics
  evaluation/  scoring saved models, paired comparison
  inference/   SentimentPredictor, preprocessing, result types
  api/         FastAPI service
  cli/         the sindhi-nlp command
  utils/       config, logging, seeding
tests/        pytest suite
docs/         API, architecture, dataset, and model card
```

## Limitations

- **One small evaluation set.** 884 test sentences from a single source, labeled by one person. Inter-annotator agreement was not measured, and "neutral" is not formally defined.
- **Short, news-style sentences.** Performance on social media, long documents, code-mixed text, other Sindhi scripts, dialects, or sarcasm is untested.
- **The synthetic data adds little.** Adding it to training raised validation macro-F1 from 0.866 to 0.887, a gap too small to treat as proven.
- **Overconfident probabilities** for the XLM-R model, as noted above.
- **Not for decisions about people** (employment, credit, moderation, legal outcomes) without human review.

See [docs/model_card.md](https://github.com/alinawazcode/sindhi-nlp/blob/main/docs/model_card.md) for the full model card.

## Development

```bash
pytest              # tests
ruff check .        # lint
ruff format --check .
mypy                # type check
```

The same checks run on every push through GitHub Actions.

## License

The code and the trained model weights are released under the MIT License (see [LICENSE](https://github.com/alinawazcode/sindhi-nlp/blob/main/LICENSE)). The human-labeled dataset is distributed separately under its own terms.

## Citation

```bibtex
@software{nawaz2026sindhinlp,
  author  = {Nawaz, Ali},
  title   = {sindhi-nlp: Sentiment classification for Sindhi text},
  year    = {2026},
  url     = {https://github.com/alinawazcode/sindhi-nlp},
  version = {0.1.1}
}
```

## Author

Ali Nawaz, alinawaz.code@gmail.com
