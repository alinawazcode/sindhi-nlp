# Choosing a model

Two models ship with the project. They answer the same question with different costs.

| | XLM-RoBERTa base (`alinawazmahar/sindhi-sentiment-xlmr`) | TF-IDF + LinearSVC (`baseline_natural`) |
| --- | --- | --- |
| Test macro-F1 | **0.8906** | 0.8695 |
| Test accuracy | 0.8903 | 0.8688 |
| Statistically better? | not shown (p = 0.116) | not shown |
| Size | about 1.1 GB | a few MB |
| Needs | PyTorch, transformers | scikit-learn only |
| Load time | seconds | milliseconds |
| Speed on a CPU | a few sentences per second | far faster |
| Training | needs a GPU (a free Kaggle T4 works) | under a second on a CPU |
| Confidence | softmax probability, overconfident | softmax of margins: a ranking score |
| Available by name on Hugging Face | yes | no, train your own |

Accuracy and F1 come from the one-time test evaluation; see [Results](../results.md).

## A practical rule

- **Start with the baseline** if you have a CPU, many documents, or tight latency. On this test set its score is within the uncertainty of XLM-R's, and you can inspect why it decides (it is a linear model over words).
- **Use XLM-R** if you can afford the memory and time, or expect text that differs from the training sentences in vocabulary, since pretrained multilingual knowledge should generalize better. That expectation is plausible but **not tested** here; measure on your own text with [`evaluate`](../tutorials/evaluate-own-data.md).
- **Do not pick on the headline number alone.** The 2-point gap is about as large as the ±2-point uncertainty of an 884-sentence test set.

## Why the baseline matters

A simple baseline protects you from fooling yourself. On the synthetic dataset every model scored 0.9998, and the baseline trained on it scored 0.60 on human-written sentences: the benchmark was solvable by memorizing templates. Always compare a transformer against a baseline trained on the same data.

## Switching models in code

Both load the same way and return the same `Prediction` type, so switching is one line:

```python
from sindhi_nlp import SentimentPredictor

fast = SentimentPredictor.from_pretrained("artifacts/models/baseline_natural")
best = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr")
```

The `confidence` values are **not comparable** between the two (`info.confidence_type` is `"score"` for the baseline and `"probability"` for XLM-R), so do not reuse a threshold across them.

## Out-of-domain warning

Both models were trained on short, news-style sentences from a single source. Social media posts, long reviews, code-mixed text, other Sindhi scripts and sarcasm are untested. See [Roadmap and known gaps](../roadmap.md).
