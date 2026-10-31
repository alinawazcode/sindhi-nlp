# Changelog

## 0.1.2 - 2026-10-31

### Added

- Documentation site (MkDocs Material): installation, quickstart, SDK, command-line and HTTP API guides, tutorials, configuration and API reference, concepts, results, FAQ and roadmap.
- `python -m sindhi_nlp.evaluation.evaluate`: score any model on any labeled CSV, list the most confident mistakes.
- `sindhi_nlp.evaluation.error_analysis` and `sindhi_nlp.evaluation.plots` (confusion matrix, model comparison, per-class F1). These modules existed but were empty.
- `sindhi-nlp --version` and `python -m sindhi_nlp`.
- `py.typed`, so type checkers use the package's type hints.
- Optional extras `plots` and `docs`.
- Docstrings on every public function and class.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and pull-request templates, Dependabot, docs-deploy and PyPI release workflows.

### Fixed

- `sindhi-nlp finalize-model` failed after a plain `pip install` because `configs/base.yaml` is not installed. It now falls back to the built-in cleaning defaults (identical to `configs/base.yaml`) and says so in `preprocessing.json`.
- A missing `configs/...` file now produces an error that explains the folder is part of the GitHub repository and is not installed by pip.

### Changed

- `evaluate_external` prints a one-line `Error:` message (exit code 1) instead of a traceback when the one-time test rule refuses a run.
- `PLAN.md` moved to `docs/archive/`.

## 0.1.1

- The README on PyPI now shows the `pip install` instructions and working links to the documentation.
- The Hugging Face upload tool now tells network problems apart from login problems, instead of reporting every failed login check as "not logged in".
- The model is named after its Hugging Face repository (not its cache folder) when loaded by name.

## 0.1.0

First release: data pipeline, TF-IDF baseline, XLM-RoBERTa training, Python SDK, command-line tool, HTTP API, and publishing tools.
