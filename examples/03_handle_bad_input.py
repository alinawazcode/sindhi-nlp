"""What happens with empty, whitespace-only or non-text input.

Run:  python examples/03_handle_bad_input.py
"""

from _model import load_predictor

from sindhi_nlp import InvalidInputError

predictor = load_predictor()
texts = ["اڄ جو ڏينهن تمام سٺو آهي", "   ", "مون کي ڏاڍو ڏک ٿيو"]

# Default: the first bad item stops the call, and the message names its position.
try:
    predictor.predict_batch(texts)
except InvalidInputError as exc:
    print("raised:", exc)  # Item 1: Text is empty after cleaning

# errors="none": good items are classified, bad items come back as None.
for text, result in zip(texts, predictor.predict_batch(texts, errors="none")):
    print(repr(text), "->", "skipped" if result is None else result.label)

# Mark uncertain results with a threshold (it never changes the label).
careful = load_predictor(confidence_threshold=0.99)
for result in careful.predict_batch(texts[:1] + texts[2:]):
    print(result.label, f"{result.confidence:.4f}", "confident" if result.is_confident else "CHECK")
