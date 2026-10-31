# Quickstart

Five minutes from install to your first predictions. It assumes you ran `pip install "sindhi-nlp[transformers]"` ([details](installation.md)).

## 1. Classify one sentence

```python
from sindhi_nlp import SentimentPredictor

predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr")

result = predictor.predict("اڄ جو ڏينهن تمام سٺو آهي")

print(result.label)  # positive
print(result.confidence)  # 0.9998998798943151
print(result.probabilities)
# {'negative': 4.56e-05, 'neutral': 5.45e-05, 'positive': 0.99990}
```

Those numbers are the real output of the published model for this sentence, run on a Windows CPU.

`predict` returns a frozen [`Prediction`](reference/python-api.md#prediction) with these fields:

| Field | Meaning |
| --- | --- |
| `text` | the text exactly as you passed it (cleaning happens internally) |
| `label` | `"negative"`, `"neutral"` or `"positive"` |
| `confidence` | the probability of the chosen label |
| `probabilities` | all three probabilities, summing to 1 |
| `is_confident` | `False` only if you set a `confidence_threshold` and the result is below it |

!!! warning
    The XLM-R model is **overconfident**: a wrong answer can still show a confidence near 1.0. Do not treat it as a calibrated probability. See [Confidence scores](concepts/confidence.md).

## 2. Classify many sentences

```python
texts = ["اڄ جو ڏينهن تمام سٺو آهي", "مون کي ڏاڍو ڏک ٿيو", "شهر ۾ بجيٽ جو اعلان ٿيو"]
for result in predictor.predict_batch(texts):
    print(result.label, f"{result.confidence:.2f}", result.text)
```

Results come back in the same order as the input. Bad items (empty, too long, not a string) raise an error by default; see [handling bad input](guide/python-sdk.md#handling-bad-input) to classify the good ones and mark the rest.

## 3. Use the command line

```bash
sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي"
```

With no `--model-dir` and no inference config, the command uses the published model automatically. The output is one line per text: label, confidence, text. Real output of the published model:

```text
$ sindhi-nlp predict "اڄ جو ڏينهن تمام سٺو آهي" "مون کي ڏاڍو ڏک ٿيو"
positive  1.000  اڄ جو ڏينهن تمام سٺو آهي
negative  1.000  مون کي ڏاڍو ڏک ٿيو
```

Before these lines, the Hugging Face libraries print progress messages such as `Fetching 8 files` and `Loading weights`; they are not part of the result (see the [FAQ](faq.md)). Note the confidence of 1.000 for both: that is the overconfidence described above.

For a CSV file:

```bash
sindhi-nlp batch --input reviews.csv --output reviews_scored.csv --text-column Text
```

See [Classify a CSV file](tutorials/classify-csv.md) for the full walk-through.

## 4. Use the baseline instead (no download, no GPU)

Train the TF-IDF baseline from your own labeled CSV in seconds, then load it the same way. See [Train on your own data](tutorials/train-own-data.md). Loading code is identical:

```python
predictor = SentimentPredictor.from_pretrained("artifacts/models/baseline_natural")
```

## Where next

- [Python SDK guide](guide/python-sdk.md): every parameter, every error, batch handling, thresholds.
- [Choosing a model](guide/choosing-a-model.md): XLM-R or baseline.
- [Evaluate on your own data](tutorials/evaluate-own-data.md): do not trust a score you did not measure on text like yours.
