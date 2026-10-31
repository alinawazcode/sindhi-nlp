"""Shared by the examples: which model to load.

By default the published model is downloaded from Hugging Face on first use
(about 1.1 GB, cached afterwards). To try an example with another model, set
the SINDHI_MODEL environment variable to a local folder or a Hub name.
"""

import os

from sindhi_nlp import SentimentPredictor

MODEL = os.environ.get("SINDHI_MODEL", "alinawazmahar/sindhi-sentiment-xlmr")


def load_predictor(**options) -> SentimentPredictor:
    """Load the model used by all examples; extra options go to ``from_pretrained``."""
    return SentimentPredictor.from_pretrained(MODEL, **options)
