"""Score a CSV file of Sindhi sentences and list the rows a person should check.

Run:  python examples/02_classify_csv.py reviews.csv reviews_scored.csv
The input needs a header row and a text column named Text (change COLUMN below).
Save the sheet as UTF-8; in Excel choose "CSV UTF-8".
"""

import sys

import pandas as pd
from _model import load_predictor

COLUMN = "Text"
REVIEW_BELOW = 0.90  # rows less confident than this go to the review list

if len(sys.argv) != 3:
    sys.exit("usage: python 02_classify_csv.py INPUT.csv OUTPUT.csv")
source, target = sys.argv[1], sys.argv[2]

df = pd.read_csv(source, encoding="utf-8-sig")  # utf-8-sig also reads files saved by Excel
if COLUMN not in df.columns:
    sys.exit(f"Column {COLUMN!r} not found. Columns in the file: {list(df.columns)}")

predictor = load_predictor()

# errors="none" gives None for blank or invalid rows instead of stopping the run
results = predictor.predict_batch(df[COLUMN].fillna("").astype(str).tolist(), errors="none")

df["predicted_label"] = [r.label if r else "" for r in results]
df["confidence"] = [round(r.confidence, 4) if r else None for r in results]

# utf-8-sig adds a byte-order mark so Excel shows Sindhi text correctly
df.to_csv(target, index=False, encoding="utf-8-sig")

print(f"Classified {sum(r is not None for r in results):,} of {len(df):,} rows -> {target}")
print(df["predicted_label"].value_counts().to_string())

# Rows worth a second look: the model was unsure, or the row could not be read.
# (XLM-R is overconfident, so a high confidence does not prove a row is right.)
low = df[(df["confidence"] < REVIEW_BELOW) | df["confidence"].isna()]
print(f"\n{len(low):,} row(s) to review (confidence below {REVIEW_BELOW} or unreadable):")
print(low.head(10).to_string())
