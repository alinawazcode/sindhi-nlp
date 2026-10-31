# Confidence scores

Every prediction carries `confidence` and `probabilities`. Understanding what they are, and are not, is the single most important thing for using the model responsibly.

## What the numbers are

| Model | How `probabilities` is computed | `ModelInfo.confidence_type` |
| --- | --- | --- |
| XLM-R | softmax of the network's raw scores (logits) | `"probability"` |
| TF-IDF + LinearSVC | softmax of the classifier's margins (no probability model is fitted) | `"score"` |

`confidence` is the largest of the three values: the model's strength of preference for its chosen label.

## Why you cannot take them literally

A **calibrated** probability means: among predictions made with 90% confidence, about 90% are right. Neither model has that property.

- XLM-R fine-tuned for a few epochs on a few thousand sentences is typically **overconfident**. Almost all predictions come out near 1.0 and wrong ones often do too. The project measured this and documents it in the model card; the confidence is not a reliable indicator of correctness.
- The baseline's value is a monotone transformation of a margin. It ranks "more certain" above "less certain" but has no frequency interpretation.

Because the softmax always sums to 1 over three classes, the minimum possible confidence is 1/3.

## What to do with them

- Use them to **rank**: in general, lower-confidence predictions are more likely to be wrong, but check this on your own data (below); for XLM-R the relationship can be weak.
- Use `confidence_threshold` to **route items to review**, not to certify accuracy. A threshold of 0.8 means "flag anything below 0.8", not "everything above it is 80% right".
- Do not compare confidences across the two models.
- If you need real probabilities, calibrate on a held-out labeled set (for example temperature scaling or isotonic regression) and check with a reliability diagram. This is on the [roadmap](../roadmap.md); the project does not ship a calibrator.

## Measure it yourself

With a labeled file ([Evaluate on your own data](../tutorials/evaluate-own-data.md)) you can check how informative confidence is for your text:

```python
import pandas as pd

bins = pd.cut(results["confidence"], [0, 0.6, 0.8, 0.95, 0.99, 1.0])
print(results.groupby(bins, observed=True)["correct"].agg(["mean", "size"]))
```

If accuracy is nearly flat across the bins, confidence is not helping you filter.
