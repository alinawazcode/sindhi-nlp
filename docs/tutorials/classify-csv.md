# Classify a CSV file

Goal: take a spreadsheet with a column of Sindhi sentences, add predicted sentiment columns, and pick out the rows a person should check.

## 1. Prepare the file

Save the sheet as CSV, UTF-8. From Excel choose *CSV UTF-8*. The file needs a header row and a text column; the default name the tool looks for is `Text` (case-insensitive).

```text
Text,Source
اڄ جو ڏينهن تمام سٺو آهي,survey
شهر ۾ بجيٽ جو اعلان ٿيو,news
```

## 2. Run the batch command

```bash
sindhi-nlp batch --input reviews.csv --output reviews_scored.csv --text-column Text
```

The first run downloads the published model (about 1.1 GB). Use `--model-dir artifacts/models/baseline_natural` for a local model, or `--device cpu` to force the CPU.

## 3. Read the result

Your columns come first, unchanged and in the same order, followed by:

| Column | Meaning |
| --- | --- |
| `predicted_label` | `negative`, `neutral` or `positive`; blank for rows that could not be classified |
| `confidence` | probability of the predicted label (see [Confidence scores](../concepts/confidence.md)) |
| `prob_negative`, `prob_neutral`, `prob_positive` | all three probabilities |
| `error` | `invalid input` for blank rows, otherwise empty |

Open it in Excel; Sindhi text displays correctly because the file is written as UTF-8 with a byte-order mark.

## 4. Review the uncertain rows

```python
import pandas as pd

df = pd.read_csv("reviews_scored.csv", encoding="utf-8-sig")

print(df["predicted_label"].value_counts(dropna=False))

# rows that could not be classified
print(df[df["error"].notna()])

# lowest-confidence rows first
uncertain = df.dropna(subset=["confidence"]).sort_values("confidence").head(50)
uncertain.to_csv("to_review.csv", index=False, encoding="utf-8-sig")
```

For XLM-R, wrong predictions can still be very confident, so reviewing only low-confidence rows will miss errors. Review a random sample of the confident rows as well, and measure the real error rate with [Evaluate on your own data](evaluate-own-data.md).

## Same thing from Python

```python
import pandas as pd
from sindhi_nlp import SentimentPredictor

predictor = SentimentPredictor.from_pretrained("alinawazmahar/sindhi-sentiment-xlmr")
df = pd.read_csv("reviews.csv", encoding="utf-8-sig")
results = predictor.predict_batch(df["Text"].astype(str).tolist(), errors="none")
df["predicted_label"] = [r.label if r else "" for r in results]
df.to_csv("reviews_scored.csv", index=False, encoding="utf-8-sig")
```

## Large files

The command loads the whole file into memory and classifies in batches of 32, which is fine for hundreds of thousands of short sentences. On a CPU, XLM-R is slow (a few sentences per second), so a very large file is better run with the baseline, or on a GPU machine.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `Column 'Text' not found; the file has [...]` | pass the real header with `--text-column` |
| garbled Sindhi in Excel | the file was saved with another encoding; re-save as CSV UTF-8 |
| `Input file not found` | paths are relative to `--project-root` (the current folder by default) |
