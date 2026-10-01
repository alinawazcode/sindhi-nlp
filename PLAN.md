# Sindhi NLP Project Plan

## Goal

Train a reproducible XLM-RoBERTa sentiment classifier for Sindhi text, package it as a Python SDK, and later expose it through a FastAPI service. The project begins with the prepared 150,000-row labeled dataset: 50,000 positive, 50,000 negative, and 50,000 neutral sentences.

## Rules To Follow

- Never change `data/raw/` after placing source data there.
- Keep notebooks for exploration only. All repeatable work must move into `src/`.
- Do not inspect the test split while selecting model settings. Use validation data for every training decision.
- Save the tokenizer, model, label mapping, configuration, metrics, and random seed together for every trained model.
- Do not commit datasets, model checkpoints, API keys, or `.env` files to GitHub.

## Dataset Split

`data/processed/train.csv`, `validation.csv`, and `test.csv` use a stratified 80/10/10 split with random seed `42`. Each file has the columns `Text` and `Label`.

| Split | Rows | Positive | Negative | Neutral | Use |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 120,000 | 40,000 | 40,000 | 40,000 | Model learning |
| Validation | 15,000 | 5,000 | 5,000 | 5,000 | Choose settings and checkpoints |
| Test | 15,000 | 5,000 | 5,000 | 5,000 | Final one-time evaluation |

## File Responsibilities

### Root Files

| File | What it will contain |
| --- | --- |
| `README.md` | Project overview, installation, quick-start commands, results, and links to the model/demo. |
| `pyproject.toml` | Package metadata, Python version, dependencies, test configuration, and CLI entry point. |
| `requirements.txt` | Pinned development and runtime dependencies for a simple local setup. |
| `.gitignore` | Rules that exclude data, checkpoints, virtual environments, cache files, and secrets. |
| `.env.example` | Names of optional environment variables without real values. |
| `Dockerfile` | Instructions to build a container for the API. |
| `docker-compose.yml` | Local multi-service setup when the API gains external services. |

### Configuration

| File | What it will contain |
| --- | --- |
| `configs/base.yaml` | Shared settings: seed, paths, label names, and output locations. |
| `configs/xlm_roberta_base.yaml` | Model name, maximum token length, batch size, learning rate, epochs, warmup, and checkpoint settings. |
| `configs/inference.yaml` | Inference-only settings such as model location, device choice, and confidence threshold. |

### Data

| File or folder | What it will contain |
| --- | --- |
| `data/raw/` | The original 50k and 100k source files, kept unchanged and ignored by Git. |
| `data/interim/` | Temporary cleaned or merged data generated during preparation. |
| `data/processed/train.csv` | The 120,000-row stratified training split. |
| `data/processed/validation.csv` | The 15,000-row validation split. |
| `data/processed/test.csv` | The 15,000-row final test split. |
| `data/processed/label_mapping.json` | The fixed mapping `negative: 0`, `neutral: 1`, `positive: 2`. |
| `data/reports/split_report.json` | Row counts, label counts, seed, and source details for the split. |
| `data/README.md` | Dataset source, license, label definitions, preparation process, and known limitations. |

### Exploration

| File | What it will contain |
| --- | --- |
| `notebooks/01_dataset_exploration.ipynb` | One-time checks for class balance, sentence lengths, missing data, duplicate analysis, and examples. |
| `notebooks/02_error_analysis.ipynb` | Analysis of incorrect model predictions after evaluation. |

### Reusable Package

| File | What it will contain |
| --- | --- |
| `src/sindhi_nlp/__init__.py` | Public package version and the small set of classes users should import. |
| `data/load_data.py` | Read CSV files and verify expected columns. |
| `data/clean_data.py` | Unicode and whitespace normalization used identically in training and inference. |
| `data/validate_data.py` | Checks for missing text, invalid labels, duplicate text, and invalid schema. |
| `data/split_data.py` | Reproducible stratified splitting from the merged clean dataset. |
| `training/train.py` | Command-line entry point that loads config and starts training. |
| `training/trainer.py` | Tokenization, Hugging Face trainer setup, checkpoint handling, and model saving. |
| `training/metrics.py` | Accuracy, precision, recall, macro-F1, and per-label metrics. |
| `training/callbacks.py` | Early stopping, best-checkpoint selection, and training logs. |
| `evaluation/evaluate.py` | Load one saved model and produce final validation or test metrics. |
| `evaluation/error_analysis.py` | Write misclassified examples and aggregate error categories. |
| `evaluation/plots.py` | Save confusion matrix and learning-curve charts. |
| `inference/predictor.py` | A stable `SentimentPredictor` interface that loads a model and returns label/confidence. |
| `inference/preprocessing.py` | Text preparation used before tokenization at prediction time. |
| `inference/schemas.py` | Typed request and prediction-result data models. |
| `api/main.py` | FastAPI application entry point. |
| `api/routes.py` | HTTP endpoints such as health check, model information, and prediction. |
| `api/dependencies.py` | Construct and cache the predictor used by API routes. |
| `cli/main.py` | Terminal commands for prediction, model information, and batch inference. |
| `utils/config.py` | Read and validate YAML configuration files. |
| `utils/logging.py` | Consistent structured logging configuration. |
| `utils/seed.py` | Set deterministic random seeds for Python, NumPy, PyTorch, and Transformers. |

### Tests, Documentation, And Automation

| File | What it will contain |
| --- | --- |
| `tests/test_clean_data.py` | Tests for text normalization and invalid text handling. |
| `tests/test_split_data.py` | Tests that splits are disjoint, reproducible, and stratified. |
| `tests/test_predictor.py` | Tests for valid prediction output and edge cases. |
| `tests/test_api.py` | Tests for the FastAPI health and prediction endpoints. |
| `docs/dataset.md` | Full dataset documentation and label policy. |
| `docs/model_card.md` | Model purpose, evaluation results, limitations, intended use, and ethical notes. |
| `docs/api.md` | Request/response examples and endpoint behavior. |
| `docs/architecture.md` | Data flow from CSV to trained model, SDK, and API. |
| `.github/workflows/tests.yml` | GitHub Actions workflow that runs formatting, tests, and type checks on each push. |

### Generated Artifacts

| Folder | What it will contain |
| --- | --- |
| `artifacts/models/` | Saved model checkpoints, tokenizer files, and label mapping copies. Store releases on Hugging Face rather than Git. |
| `artifacts/metrics/` | JSON metrics, confusion matrices, and learning curves. |
| `artifacts/reports/` | A concise training summary for every important experiment. |

## Implementation Order

1. Review the data split and finish `data/README.md`.
2. Implement and test data validation, cleaning, and configuration loading.
3. Fill the XLM-RoBERTa configuration and implement training with macro-F1 as the key metric.
4. Train a small test run first, then complete the full run.
5. Evaluate the best checkpoint once on `test.csv` and create the model card.
6. Build the `SentimentPredictor` SDK interface and tests.
7. Add the FastAPI service, Docker support, CLI, and GitHub Actions.
8. Publish the model to Hugging Face and the package to PyPI after the API and tests are stable.
