# Command line

Installing the package adds the `sindhi-nlp` command. `python -m sindhi_nlp` does the same thing, which is handy when the scripts folder is not on your `PATH`.

```text
sindhi-nlp [-h] [--version] [--log-level LOG_LEVEL] {predict,batch,info,finalize-model} ...
```

| Global option | Meaning |
| --- | --- |
| `--version` | print `sindhi-nlp <version>` and exit (added after 0.1.1; see [Which version has what](../installation.md#which-version-has-what)) |
| `--log-level` | logging level on stderr, e.g. `DEBUG`, `INFO`, `WARNING` (default) |

## Which model is used?

The `predict`, `batch` and `info` commands share these options:

| Option | Meaning |
| --- | --- |
| `--model-dir PATH_OR_NAME` | a local model folder or a Hugging Face name such as `alinawazmahar/sindhi-sentiment-xlmr` |
| `--config FILE` | inference config (default `configs/inference.yaml`) |
| `--project-root DIR` | folder that `--model-dir`, `--config`, `--input` and `--output` are resolved against (default: the current folder) |
| `--device {auto,cpu,cuda}` | where to run the model |

The model is chosen in this order:

1. `--model-dir`, if given. A local folder wins; otherwise it is treated as a Hugging Face name.
2. The `model_dir` in the inference config, if the config file exists.
3. Otherwise the published model `alinawazmahar/sindhi-sentiment-xlmr` (a message on stderr says so; the first use downloads about 1.1 GB).

After `pip install sindhi-nlp` in an ordinary folder there is no `configs/inference.yaml`, so step 3 applies. Inside a clone of the repository the config points at `artifacts/models/xlmr_mixed`, so pass `--model-dir` unless you have that folder.

## `predict`

Classify one or more texts given on the command line.

```text
sindhi-nlp predict [--json] [model options] texts [texts ...]
```

```bash
sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي" "مون کي ڏاڍو ڏک ٿيو"
```

Default output is a table with the label, the confidence and the text, one row per input. Real output of the published model:

```text
positive  1.000  اڄ جو ڏينهن تمام سٺو آهي
negative  1.000  مون کي ڏاڍو ڏک ٿيو
```

With `--json` it prints a JSON list, one object per text (the same fields as `Prediction.to_dict()`). Real output of the published model:

```json
[
  {
    "text": "اڄ جو ڏينهن تمام سٺو آهي",
    "label": "positive",
    "confidence": 0.9998998798943151,
    "probabilities": {
      "negative": 4.5615245206568106e-05,
      "neutral": 5.45048604784391e-05,
      "positive": 0.9998998798943151
    },
    "is_confident": true
  }
]
```

Other things to know:

- A text that cannot be classified (for example only spaces) does not stop the others. Its row prints as `invalid  -  '<text>'` (and as `null` in `--json` output), and the command exits with code 1 once everything has been printed.
- If you set a `confidence_threshold` in the inference config, results below it get a `(low confidence)` suffix.

!!! note "PowerShell and Sindhi text"
    Sindhi text printed correctly in the author's Windows test. If yours looks wrong (for example `????`), the console code page is the likely cause: run `chcp 65001` first, or put the texts in a UTF-8 file and use `batch`.

## `batch`

Classify every row of a CSV (or every line of a text file) and write a CSV.

```text
sindhi-nlp batch --input FILE --output FILE [--text-column NAME] [model options]
```

| Option | Meaning |
| --- | --- |
| `--input` | a `.csv` file, or any other file read as text with one sentence per line (the rows then go into a column named by `--text-column`) |
| `--output` | CSV to write. Missing folders are created. |
| `--text-column` | the column holding the text (CSV input). Default `Text`, matched case-insensitively. A missing column is an error that lists the columns the file has. |

```bash
sindhi-nlp batch --input reviews.csv --output reviews_scored.csv --text-column Text
```

The output keeps your rows in order and adds columns (real output from a small toy model, to show the layout):

```text
Text,predicted_label,confidence,prob_negative,prob_neutral,prob_positive,error
سٺو شاندار خوش,positive,0.778694,0.110653,0.110653,0.778694,
خراب ڏکيو بڪار,negative,0.778694,0.778694,0.110653,0.110653,
,,,,,,invalid input
شهر ڪتاب ميز,neutral,0.778694,0.110653,0.778694,0.110653,
```

- Empty or invalid rows are **not** dropped. They stay in place, the prediction columns are blank, and `error` says `invalid input`.
- The command prints a summary such as `Classified 3 of 4 rows -> out.csv` and, if any rows failed, how many.
- Output is written as UTF-8 with a byte-order mark so Excel opens Sindhi text correctly.

## `info`

Print details of a model without classifying anything.

```bash
sindhi-nlp info --model-dir artifacts/models/baseline_natural
sindhi-nlp info --model-dir alinawazmahar/sindhi-sentiment-xlmr --json
```

Real output for the published model:

```text
name                   alinawazmahar/sindhi-sentiment-xlmr
backend                transformers
model_dir              C:\Users\<you>\.cache\huggingface\hub\models--alinawazmahar--sindhi-sentiment-xlmr\snapshots\7665f1262b59...
labels                 ['negative', 'neutral', 'positive']
confidence_type        probability
device                 cpu
max_length             128
preprocessing_source   preprocessing.json
seed                   42
base_model             xlm-roberta-base
```

`model_dir` is the local cache folder the download went to. For the TF-IDF baseline, `backend` is `sklearn`, `confidence_type` is `score`, and `max_length` and `base_model` are empty.

## `finalize-model`

Writes `preprocessing.json` next to a model you trained, recording the cleaning options it was trained with, so prediction reproduces them. Run it once per trained model, before using or publishing it.

```bash
sindhi-nlp finalize-model --model-dir artifacts/models/xlmr_mixed
sindhi-nlp finalize-model --model-dir artifacts/models/xlmr_mixed --base-config configs/base.yaml
```

Without this file, a model falls back to the default cleaning options and `info` shows `preprocessing_source: defaults (no preprocessing.json found)`. The default matches `configs/base.yaml`, so this only matters if you changed the cleaning settings. See [Text preprocessing](preprocessing.md).

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | a handled error (missing file, bad column, model not found; the message starts with `Error:` on stderr), or `predict` was given at least one invalid text |
| 2 | wrong command-line usage (reported by `argparse`) |

## Other command-line tools

The `python -m` modules below are for the training and evaluation workflow. Each accepts `--help`.

| Module | Purpose | Guide |
| --- | --- | --- |
| `sindhi_nlp.evaluation.evaluate` | score any model on any labeled CSV | [Evaluation tools](../reference/evaluation.md#evaluate) |
| `sindhi_nlp.evaluation.evaluate_external` | score on the project's saved splits (test split is one-time) | [Evaluation tools](../reference/evaluation.md#evaluate_external) |
| `sindhi_nlp.evaluation.compare_models` | paired significance test between two models | [Evaluation tools](../reference/evaluation.md#compare_models) |
| `sindhi_nlp.training.baseline` | train the TF-IDF baseline | [Training](../tutorials/train-own-data.md) |
| `sindhi_nlp.training.train` | fine-tune XLM-R | [Training](../tutorials/train-own-data.md) |
| `sindhi_nlp.data.split_data`, `prepare_external` | build train/validation/test files | [Data pipeline](../reference/data-pipeline.md) |
| `sindhi_nlp.cli.publish_model` | upload a model to Hugging Face | [Publishing](../publishing.md) |
