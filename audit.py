import re
import pandas as pd

days = ["اربع", "خميس", "اڱاري", "آچر", "جمعي", "سومر", "ڇنڇر"]
pat = re.compile("|".join(days))
mask = lambda s: pat.sub("<DAY>", s)

tr = pd.read_csv("data/processed/train.csv")
va = pd.read_csv("data/processed/validation.csv")
tr_sk = set(tr.Text.map(mask))
va["sk"] = va.Text.map(mask)
print("validation rows whose weekday-masked twin is in train:", round(va.sk.isin(tr_sk).mean(), 4))