import re
import pandas as pd

tr = pd.read_csv("data/processed/train.csv")
va = pd.read_csv("data/processed/validation.csv")
norm = lambda s: re.sub(r"[\W\d_]+", " ", s).strip()

va["n"] = va.Text.map(norm)
print("validation texts matching a train text after stripping punctuation/digits:",
      round(va.n.isin(set(tr.Text.map(norm))).mean(), 4))

print(tr.assign(words=tr.Text.str.split().str.len())
        .groupby("Label").words.agg(["mean", "std", "min", "max"]))

print("\nMost common first three words (template check):")
print(tr.Text.map(norm).str.split().str[:3].str.join(" ").value_counts().head(10))