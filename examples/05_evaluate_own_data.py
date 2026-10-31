"""Measure the model on your own labeled sentences, and see its mistakes.

Run:  python examples/05_evaluate_own_data.py my_labeled.csv
The CSV needs a text column (Text) and a label column (Label) holding
negative, neutral or positive. Needs: pip install "sindhi-nlp[plots,transformers]"
"""

import sys

import pandas as pd
from _model import load_predictor

from sindhi_nlp.evaluation.error_analysis import confusion_pairs, misclassified
from sindhi_nlp.evaluation.evaluate import evaluate_predictor, format_report

if len(sys.argv) != 2:
    sys.exit("usage: python 05_evaluate_own_data.py LABELED.csv")

df = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
predictor = load_predictor()

metrics, results = evaluate_predictor(predictor, df["Text"].astype(str).tolist(), df["Label"].astype(str).tolist())
print(format_report(metrics, title="My data"))

# Which mix-ups happen most, and the most confident mistakes first
print("\nMost common mix-ups:")
print(confusion_pairs(results).head(5).to_string(index=False))
print("\nMost confident mistakes:")
print(misclassified(results, limit=10).to_string(index=False))

# Optional picture (needs the plots extra):
#   from sindhi_nlp.evaluation.plots import plot_confusion_matrix
#   plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], path="confusion.png")
