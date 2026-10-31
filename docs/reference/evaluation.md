# Evaluation tools

!!! info "Not in the 0.1.1 release on PyPI"
    The `evaluate` tool, the error-analysis helpers and the plotting functions were added after 0.1.1. `evaluate_external` and `compare_models` are in 0.1.1. Until the next release, install from GitHub: `pip install "sindhi-nlp[plots,transformers] @ git+https://github.com/alinawazcode/sindhi-nlp.git"`, or clone the repository and run `pip install -e .`. See [Which version has what](../installation.md#which-version-has-what).


Three tools, for three jobs.

| Tool | Use it to |
| --- | --- |
| [`evaluate`](#evaluate) | score **any** model on **any** labeled CSV |
| [`evaluate_external`](#evaluate_external) | score a model on the project's saved splits, with the one-time test rule |
| [`compare_models`](#compare_models) | test whether two scored models really differ |

## evaluate

```text
python -m sindhi_nlp.evaluation.evaluate --model-dir MODEL --csv FILE
    [--text-column Text] [--label-column Label] [--device auto|cpu|cuda]
    [--output FILE.json] [--show-errors N]
```

| Option | Meaning |
| --- | --- |
| `--model-dir` | local folder or Hugging Face name |
| `--csv` | CSV with a text column and a label column |
| `--text-column`, `--label-column` | header names (defaults `Text`, `Label`; must match exactly) |
| `--output` | write metrics JSON here, and the mistakes to `<name>_misclassified.csv` beside it |
| `--show-errors N` | print the N most confident mistakes |

Labels are lower-cased and stripped before comparing. Exit code 1 and an `Error:` line for a missing column, unknown gold label, unreadable file or missing model. Python equivalent: `evaluate_predictor(predictor, texts, y_true)`.

The metrics JSON has `accuracy`, `macro_f1`, `macro_precision`, `macro_recall`, `per_label` (precision, recall, f1, support), `confusion_matrix` (rows true, columns predicted, in `labels` order), `num_examples` and `skipped`.

The mistakes file has columns `text`, `true`, `predicted`, `confidence`, `correct`, most confident mistake first.

## evaluate_external

```text
python -m sindhi_nlp.evaluation.evaluate_external --model-dir artifacts/models/NAME
    [--split train|validation|test] [--data FILE] [--tag external]
    [--config configs/base.yaml] [--project-root .] [--final] [--overwrite]
```

Scores a saved model on `data/external/processed/<split>.csv` (or `--data`) and writes:

- `artifacts/metrics/<model>_<tag>_<split>.json`
- `artifacts/reports/<model>_<tag>_<split>_misclassified.csv`

Rules for the test split: `--final` is required; a second run for the same model and tag is refused unless `--overwrite`. Validation can be repeated freely. Errors print `Error: ...` with exit code 1.

## compare_models

```text
python -m sindhi_nlp.evaluation.compare_models --a NAME_A --b NAME_B
    [--split validation|test] [--tag external] [--config ...] [--project-root .]
```

Compares two models that were both scored by `evaluate_external` on the same split. It reads the saved metrics and mistake lists, so it loads no models and cannot start a new test run. It reports each model's accuracy with a 95% Wilson interval, how the models agree (both right, both wrong, only A wrong, only B wrong), and the exact McNemar p-value, and saves `artifacts/metrics/compare_<A>_vs_<B>_<tag>_<split>.json`.

!!! note
    `compare_models` needs the files written by `evaluate_external`. The mistake file written by `evaluate` has a different layout and is not accepted.

Reading the result:

- p below 0.05: the difference is unlikely to be chance.
- p above 0.05 (for the published pair, 0.1155): not shown to differ. Prefer the cheaper model.

See [Evaluation method](../concepts/evaluation-method.md).

## Error analysis helpers

```python
from sindhi_nlp.evaluation.error_analysis import build_results_frame, misclassified, confusion_pairs

frame = build_results_frame(texts, y_true, y_pred, confidences)
misclassified(frame, limit=20)  # most confident mistakes first
confusion_pairs(frame)  # which (true, predicted) mistakes are most common
```

## Plots

Need `pip install "sindhi-nlp[plots]"`. They use matplotlib's object API, so they are safe in servers and notebooks and never open a window.

```python
from sindhi_nlp.evaluation import plots

fig = plots.plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], path="cm.png", normalize=True)
plots.plot_model_comparison({"baseline": 0.8695, "xlmr": 0.8906}, path="cmp.png", metric="Test macro-F1")
plots.plot_per_label_f1(metrics["per_label"], path="f1.png")
```

`normalize=True` shows row percentages (recall per class). The comparison chart's axis starts at zero so small differences are not exaggerated. The figures on the [Results](../results.md) page were made with these functions from the saved metrics.
