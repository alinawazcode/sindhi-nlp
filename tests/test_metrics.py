import numpy as np
import pytest

from sindhi_nlp.training.metrics import compute_metrics, make_hf_compute_metrics

LABELS = ["negative", "neutral", "positive"]


def test_perfect_predictions():
    y = ["negative", "neutral", "positive", "neutral"]
    m = compute_metrics(y, y, LABELS)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]


def test_known_values_and_matrix_orientation():
    y_true = ["negative", "negative", "neutral", "positive"]
    y_pred = ["negative", "neutral", "neutral", "neutral"]
    m = compute_metrics(y_true, y_pred, LABELS)
    # rows = true, columns = predicted
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    assert m["accuracy"] == 0.5
    assert m["per_label"]["negative"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert m["per_label"]["neutral"]["precision"] == pytest.approx(1 / 3)
    assert m["per_label"]["positive"]["f1"] == 0.0  # never predicted: zero, not an error
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 0.0) / 3)


def test_invalid_inputs():
    with pytest.raises(ValueError, match="length"):
        compute_metrics(["a"], ["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="zero examples"):
        compute_metrics([], [], ["a"])


def test_hf_compute_metrics_from_logits():
    compute = make_hf_compute_metrics(LABELS)
    logits = np.array([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0], [3.0, 0, 0]])
    result = compute((logits, np.array([0, 1, 2, 1])))
    assert result["accuracy"] == 0.75
    assert result["f1_positive"] == 1.0
    assert set(result) >= {"macro_f1", "macro_precision", "macro_recall", "f1_negative", "f1_neutral"}
    # Models that return a tuple of outputs are handled too.
    assert compute(((logits, None), np.array([0, 1, 2, 0])))["accuracy"] == 1.0