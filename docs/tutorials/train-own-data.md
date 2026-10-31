# Train on your own data

This tutorial takes a labeled CSV to a trained, evaluated model. The baseline needs a CPU and seconds. Fine-tuning XLM-R needs a GPU.

Run everything from a **project folder** that contains `configs/` (clone the repository, or copy its `configs/` folder). The commands use paths relative to it, and the package must be importable (`pip install -e .` in the clone).

## 1. Put your data in place

You need a CSV with a text column and a label column. Labels are `negative`, `neutral`, `positive` in any capitalization.

Copy it to `data/external/natural_sindhi.csv` (the default) and tell the pipeline your column names in `configs/external.yaml`:

```yaml
source_file: data/external/natural_sindhi.csv
text_column: Sindhi Text      # header names in YOUR file, matched case-insensitively
label_column: Sentiment
duplicate_policy: drop
split:
  train: 0.7
  validation: 0.1
  test: 0.2
output_dir: data/external/processed
```

Do not put it in `data/raw/`; that folder is reserved for the synthetic data and the tool refuses it.

## 2. Build the splits

```bash
python -m sindhi_nlp.data.prepare_external
```

Real output (a toy dataset of 360 rows):

```text
Rows read: 360   kept: 360
Labels: {'negative': 120, 'neutral': 120, 'positive': 120}
  train         252  {'negative': 84, 'neutral': 84, 'positive': 84}
  validation     36  {'negative': 12, 'neutral': 12, 'positive': 12}
  test           72  {'negative': 24, 'neutral': 24, 'positive': 24}
Words per text (min/mean/max): 5/5.0/5   vocabulary: 372
```

It writes `train.csv`, `validation.csv`, `test.csv` (columns `Text`, `Label`) and `split_report.json` (counts, seed, file hashes, cleaning options) into `data/external/processed/`. The step cleans text, drops rows with missing text or unknown labels, removes duplicates, keeps near-duplicates in the same split, and is repeatable: the same input gives byte-identical files.

!!! danger "Protect the test split"
    `test.csv` is for **one** final evaluation per model. Make every choice (features, thresholds, epochs, which model) on the validation split. If you peek at the test split repeatedly, your final number is no longer an honest estimate.

## 3. Train the baseline

```bash
python -m sindhi_nlp.training.baseline --data-dir data/external/processed --model-name my_baseline
```

```text
Loaded 252 train and 36 validation rows
Trained in 0.1s
Validation macro-F1: 1.0000   accuracy: 1.0000
  negative   F1 1.0000
  neutral    F1 1.0000
  positive   F1 1.0000
```

(The perfect score is because the toy data is trivially separable; yours will not be.)

It creates:

| Path | Contents |
| --- | --- |
| `artifacts/models/my_baseline/` | `model.joblib`, `config.json`, `label_mapping.json`, `metrics.json` |
| `artifacts/metrics/my_baseline_validation.json` | validation metrics |
| `artifacts/reports/my_baseline_summary.md` | readable summary with the strongest words per class |
| `artifacts/reports/my_baseline_misclassified_validation.csv` | validation mistakes |

Settings (n-grams, `min_df`, `C`) are in `configs/baseline.yaml`; see [Configuration files](../reference/configuration.md#baseline-yaml).

## 4. Save the cleaning settings with the model

```bash
sindhi-nlp finalize-model --model-dir artifacts/models/my_baseline
```

This writes `preprocessing.json` so prediction cleans text exactly as training did ([why](../guide/preprocessing.md)).

## 5. Evaluate

Validation can be run as often as you like:

```bash
python -m sindhi_nlp.evaluation.evaluate_external --model-dir artifacts/models/my_baseline --split validation
```

When you are done choosing, score the test split once:

```bash
python -m sindhi_nlp.evaluation.evaluate_external --model-dir artifacts/models/my_baseline --split test --final
```

Without `--final` the tool refuses; a second run for the same model is refused too unless you add `--overwrite`. Both refusals are deliberate and print a one-line `Error:` message with exit code 1.

## 6. Use the model

```python
from sindhi_nlp import SentimentPredictor

predictor = SentimentPredictor.from_pretrained("artifacts/models/my_baseline")
print(predictor.predict("اڄ جو ڏينهن تمام سٺو آهي").label)
```

## 7. Fine-tune XLM-R (GPU)

Copy `configs/xlm_roberta_base.yaml` and edit it to use only your data:

```yaml
model_name: xlm-roberta-base
run_name: my_xlmr                 # output folder: artifacts/models/my_xlmr

datasets:
  mine: data/external/processed
train_on: [mine]
validate_on: [mine]              # the checkpoint is chosen on this validation split
report_on: []

tokenizer:
  max_length: 128

training:
  learning_rate: 2.0e-5
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 64
  num_train_epochs: 4
  warmup_ratio: 0.1
  weight_decay: 0.01
  early_stopping_patience: 2
  fp16: auto
```

First check that everything works with a tiny run:

```bash
python -m sindhi_nlp.training.train --config configs/my_xlmr.yaml --smoke
```

Then train:

```bash
python -m sindhi_nlp.training.train --config configs/my_xlmr.yaml
```

Training reads only `train.csv` and `validation.csv`, never the test split. It picks the best epoch by validation macro-F1, writes the model and tokenizer, `training_config.json` (settings, seed, data hashes, library versions) and metrics to `artifacts/models/my_xlmr/`. It refuses to overwrite an existing model folder unless you pass `--overwrite`.

A free Kaggle T4 trains this in minutes for a few thousand sentences. On a CPU it is impractically slow; use the baseline, or train on a GPU machine and copy the model folder back. Then run `finalize-model`, evaluate, and compare to the baseline:

```bash
python -m sindhi_nlp.evaluation.compare_models --a my_baseline --b my_xlmr --split validation --tag external
```

See [Evaluation tools](../reference/evaluation.md#compare_models) for how to read the result.

!!! note "Reproducing the published model"
    `configs/xlmr_mixed.yaml` holds the exact settings of the published model (3 epochs, a 30,000-row sample of the synthetic data, and the human-labeled data repeated 5 times). It needs both datasets in place. The default `configs/xlm_roberta_base.yaml` is a general starting point and trains for up to 4 epochs.

## Publishing your model

See [Publishing a release](../publishing.md) for the Hugging Face upload tool (dry run by default, private first).
