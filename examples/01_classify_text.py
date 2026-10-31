"""Classify one sentence, then several, and inspect the model.

Run:  python examples/01_classify_text.py
"""

from _model import load_predictor

predictor = load_predictor()

# One sentence ------------------------------------------------------------
result = predictor.predict("اڄ جو ڏينهن تمام سٺو آهي")
print(result.label)  # positive
print(f"{result.confidence:.3f}")  # probability of that label
print(result.probabilities)  # all three probabilities
print(result.to_dict())  # plain dict, ready for json.dumps(..., ensure_ascii=False)

# Several sentences (same order as the input) --------------------------------
texts = [
    "اڄ جو ڏينهن تمام سٺو آهي",
    "مون کي ڏاڍو ڏک ٿيو",
    "شهر ۾ بجيٽ جو اعلان ٿيو",
]
for item in predictor.predict_batch(texts):
    print(f"{item.label:<9} {item.confidence:.2f}  {item.text}")

# About the model ---------------------------------------------------------
info = predictor.info
print(info.name, info.backend, info.labels, info.device)
