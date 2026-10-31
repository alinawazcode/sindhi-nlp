# Evaluate on your own data

!!! info "Not in the 0.1.1 release on PyPI"
    The `evaluate` command and the plots used here were added after 0.1.1. Until the next release, install from GitHub: `pip install "sindhi-nlp[plots,transformers] @ git+https://github.com/alinawazcode/sindhi-nlp.git"`, or clone the repository and run `pip install -e .`. See [Which version has what](../installation.md#which-version-has-what).


A published score describes someone else's sentences. Before relying on a model, label a few hundred sentences from your own source and measure.

## 1. Build a labeled file

A CSV with two columns, the text and the label. Labels must be exactly the model's labels (`negative`, `neutral`, `positive`); the tool lower-cases and trims them, so `Positive ` also works.

```text
Text,Label
اڄ جو ڏينهن تمام سٺو آهي,positive
مون کي ڏاڍو ڏک ٿيو,negative
```

Guidance:

- **How many?** At least 300, ideally more than 1,000. With 884 sentences the 95% uncertainty on accuracy is about ±2 points; with 100 it is about ±6.
- **Sample randomly** from the text you will really classify, not from the examples you find easiest to label.
- **Do not reuse** sentences the model was trained on, or your numbers will be too optimistic.
- Label without looking at the model's predictions.

## 2. Score it

```bash
python -m sindhi_nlp.evaluation.evaluate \
    --model-dir alinawazmahar/sindhi-sentiment-xlmr \
    --csv my_labeled.csv \
    --text-column Text --label-column Label \
    --output results/my_eval.json \
    --show-errors 10
```

Real output (toy model on its own validation file, to show the format):

```text
my_baseline on validation.csv
36 examples   accuracy 1.0000   macro-F1 1.0000
label        precision    recall        f1  support
negative         1.000     1.000     1.000       12
neutral          1.000     1.000     1.000       12
positive         1.000     1.000     1.000       12

confusion matrix (rows = true, columns = predicted)
            negative   neutral  positive
negative          12         0         0
neutral            0        12         0
positive           0         0        12

most confident mistakes (0 in total)
```

With mistakes present, the report also prints `most confident mistakes` (the `--show-errors N` rows, most confident first) and `most common mistakes` (for example `negative predicted as positive: 23`).

`--output` writes two files: `results/my_eval.json` (all metrics) and `results/my_eval_misclassified.csv` (every mistake with its confidence, most confident first).

Rows the model refuses (empty after cleaning, too long) are skipped and counted in `skipped` in the JSON, instead of failing the run.

## 3. Read the numbers

- **Accuracy** is the share of sentences right. It misleads when classes are unbalanced.
- **Macro-F1** averages the per-class F1, weighting each class equally. It is the number this project reports.
- **Per-class precision / recall** show which class hurts. Low neutral recall means neutral sentences are being called positive or negative.
- **The confusion matrix** shows which confusions happen. Rows are the truth, columns the prediction.

## 4. Look at the mistakes

```python
import pandas as pd

wrong = pd.read_csv("results/my_eval_misclassified.csv", encoding="utf-8-sig")
print(wrong.head(20)[["text", "true", "predicted", "confidence"]])
```

Look for patterns: a topic the model misreads, a label you now disagree with (fix your data), a text style the training data lacked. Confident mistakes are the most informative; they are exactly what a confidence filter would not catch.

## 5. Compare two models fairly

Score both on the **same** file and compare, using a paired test rather than eyeballing two numbers. The project's `compare_models` tool does this for the project's saved splits ([Evaluation tools](../reference/evaluation.md#compare_models)); for your own file, use the same idea from Python:

```python
from sindhi_nlp.evaluation.evaluate import evaluate_predictor

metrics_a, results_a = evaluate_predictor(model_a, texts, gold)
metrics_b, results_b = evaluate_predictor(model_b, texts, gold)
print(metrics_a["macro_f1"], metrics_b["macro_f1"])
```

If the models differ by less than the uncertainty of your test set, you cannot say one is better.

## 6. Draw it

With `pip install "sindhi-nlp[plots]"`:

```python
from sindhi_nlp.evaluation import plots

plots.plot_confusion_matrix(metrics_b["confusion_matrix"], metrics_b["labels"], path="confusion.png", title="My data")
plots.plot_per_label_f1(metrics_b["per_label"], path="f1.png")
```

## From Python

```python
import pandas as pd
from sindhi_nlp import SentimentPredictor
from sindhi_nlp.evaluation.evaluate import evaluate_predictor, format_report

predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr")
df = pd.read_csv("my_labeled.csv", encoding="utf-8-sig")
metrics, results = evaluate_predictor(
    predictor, df["Text"].astype(str).tolist(), df["Label"].str.strip().str.lower().tolist()
)
print(format_report(metrics, "my data"))
```
