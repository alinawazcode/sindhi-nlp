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

## Project Phases

### Phase 0: Repository Setup

**Goal:** Create a clean, repeatable project workspace before writing ML code.

**Work:** Confirm the folders, Git ignore rules, Python version, dependency files, and project plan. Keep the raw data and trained-model artifacts out of GitHub.

**Finish when:** A clean clone contains the project structure, `PLAN.md`, and no large dataset/model files are tracked by Git.

### Phase 1: Data Audit And Documentation

**Goal:** Prove that the training data is usable and document what it represents.

**Work:** Inspect labels, empty values, duplicate sentences, sentence lengths, and examples from each class. Document the data source, label meanings, language limitations, and any synthetic or manually labeled content in `data/README.md` and `docs/dataset.md`.

**Finish when:** You can explain where all 150,000 rows came from, what each label means, and why the data is appropriate for sentiment classification.

### Phase 2: Reproducible Data Pipeline

**Goal:** Make cleaning and splitting repeatable from raw input files.

**Work:** Implement `load_data.py`, `clean_data.py`, `validate_data.py`, and `split_data.py`. The pipeline must normalize text, validate the three labels, detect exact duplicate text, and create the same stratified 80/10/10 split from the same seed.

**Finish when:** A single command recreates `train.csv`, `validation.csv`, `test.csv`, `label_mapping.json`, and `split_report.json` with identical row counts.

### Phase 3: Baseline Model

**Goal:** Establish a simple reference score before using deep learning.

**Work:** Train a TF-IDF plus LinearSVC baseline using only training data. Record validation macro-F1, per-class F1, training time, and common mistakes.

**Finish when:** You have a saved baseline metrics report. Do not optimize this model heavily; it is the comparison point for XLM-RoBERTa.

### Phase 4: XLM-RoBERTa Training Pipeline

**Goal:** Fine-tune a multilingual transformer reproducibly.

**Work:** Complete `configs/xlm_roberta_base.yaml`, `train.py`, `trainer.py`, `metrics.py`, `callbacks.py`, `config.py`, and `seed.py`. Start with a small run to verify GPU/CPU setup, saving, evaluation, and logging. Then run the full training job using the training split and choose the best checkpoint by validation macro-F1.

**Finish when:** The model, tokenizer, config, label mapping, seed, and validation metrics are saved together in one versioned model folder.

### Phase 5: Evaluation And Error Analysis

**Goal:** Measure the model honestly and understand its weaknesses.

**Work:** Run the chosen checkpoint once on the untouched test split. Generate accuracy, macro-F1, precision/recall per class, confusion matrix, and a table of misclassified examples. Compare results against the baseline.

**Finish when:** `docs/model_card.md` reports test results, limitations, likely failure cases, hardware used, and the correct intended use of the model.

### Phase 6: Python SDK

**Goal:** Make the trained model easy for another Python developer to use.

**Work:** Implement `SentimentPredictor` in `inference/predictor.py`, shared preprocessing, typed result schemas, and a small CLI. The public interface should accept text and return `label`, `confidence`, and optional class probabilities.

**Finish when:** A developer can install the package locally and make a prediction with a few lines of Python or one terminal command.

### Phase 7: API Service

**Goal:** Make the model available to web and mobile applications.

**Work:** Implement FastAPI entry points, health checks, prediction routes, request validation, structured error responses, and cached model loading. Write API documentation with example requests and responses.

**Finish when:** The service starts locally, `GET /health` reports the loaded model version, and `POST /predict` returns validated predictions.

### Phase 8: Testing, Packaging, And Automation

**Goal:** Make the project safe to change and professional to share.

**Work:** Fill every test file, configure formatting/type checking, write GitHub Actions tests, complete `pyproject.toml`, and build a Docker image. Test malformed text, empty input, missing model files, consistent label mapping, and API responses.

**Finish when:** A clean environment can install the package, run tests, build the container, and start the API without manual fixes.

### Phase 9: Publish And Present

**Goal:** Turn the finished work into a strong public portfolio project.

**Work:** Publish the model to Hugging Face, publish the Python package to PyPI only after versioning is stable, add screenshots/results to the README, and release version `0.1.0`. Keep the repository focused on code, docs, metrics, and small examples rather than the full dataset.

**Finish when:** A visitor can understand the project, reproduce the results, download the model, install the package, and try the API from the README.
