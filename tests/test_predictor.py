"""Tests for the prediction interface (scikit-learn backend plus a fake transformers backend)."""
import json
import math

import joblib
import numpy as np
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import sindhi_nlp
from sindhi_nlp.data.clean_data import CleaningOptions
from sindhi_nlp.inference.predictor import Backend, SentimentPredictor, softmax
from sindhi_nlp.inference.preprocessing import TextPreprocessor, write_preprocessing_config
from sindhi_nlp.inference.schemas import (
    InvalidInputError,
    ModelInfo,
    ModelLoadError,
    ModelNotFoundError,
    Prediction,
)

TRAIN = (
    [("خراب ڏکيو بڪار", "negative")] * 20
    + [("شهر ڪتاب ميز", "neutral")] * 20
    + [("سٺو شاندار خوش", "positive")] * 20
)


@pytest.fixture
def model_dir(tmp_path):
    texts, labels = zip(*TRAIN)
    pipeline = Pipeline([("tfidf", TfidfVectorizer(token_pattern=r"\S+")), ("svc", LinearSVC(random_state=0))])
    pipeline.fit(list(texts), list(labels))
    directory = tmp_path / "models" / "toy"
    directory.mkdir(parents=True)
    joblib.dump(pipeline, directory / "model.joblib")
    (directory / "config.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
    return directory


@pytest.fixture
def predictor(model_dir):
    return SentimentPredictor.from_pretrained(model_dir)


# ------------------------------------------------------------------ predictions

def test_predict_returns_a_valid_result(predictor):
    result = predictor.predict("سٺو شاندار خوش")
    assert isinstance(result, Prediction)
    assert result.label == "positive"
    assert result.text == "سٺو شاندار خوش"
    assert set(result.probabilities) == {"negative", "neutral", "positive"}
    assert math.isclose(sum(result.probabilities.values()), 1.0)
    assert result.confidence == max(result.probabilities.values())
    assert 1 / 3 <= result.confidence <= 1.0 and result.is_confident


def test_each_class_is_recognized(predictor):
    assert predictor.predict("خراب ڏکيو").label == "negative"
    assert predictor.predict("شهر ميز").label == "neutral"


def test_prediction_is_json_serializable(predictor):
    payload = json.dumps(predictor.predict("سٺو").to_dict(), ensure_ascii=False)
    assert json.loads(payload)["label"] == "positive"


def test_batch_keeps_order_and_matches_single_predictions(predictor):
    texts = ["سٺو شاندار", "خراب ڏکيو", "شهر ڪتاب", "سٺو خوش", "خراب بڪار"]
    batch = predictor.predict_batch(texts)
    assert [r.label for r in batch] == ["positive", "negative", "neutral", "positive", "negative"]
    for text, result in zip(texts, batch):
        assert result == predictor.predict(text)
    assert predictor.predict_batch(texts, batch_size=2) == batch  # chunking changes nothing


def test_empty_batch_and_string_misuse(predictor):
    assert predictor.predict_batch([]) == []
    with pytest.raises(TypeError, match="list of texts"):
        predictor.predict_batch("a single string")
    with pytest.raises(ValueError):
        predictor.predict_batch(["x"], batch_size=0)
    with pytest.raises(ValueError):
        predictor.predict_batch(["x"], errors="ignore")


# ------------------------------------------------------------------ bad input

@pytest.mark.parametrize("bad", ["", "   ", "\u200b\u200f", None, 5, ["a"], b"bytes"])
def test_invalid_input_raises(predictor, bad):
    with pytest.raises(InvalidInputError):
        predictor.predict(bad)


def test_overlong_text_is_rejected(model_dir):
    short = SentimentPredictor.from_pretrained(model_dir, max_chars=20)
    with pytest.raises(InvalidInputError, match="limit"):
        short.predict("سٺو " * 30)
    assert short.predict("سٺو").label == "positive"


def test_batch_error_names_the_bad_item(predictor):
    with pytest.raises(InvalidInputError, match="Item 1"):
        predictor.predict_batch(["سٺو", "", "خراب"])


def test_batch_errors_none_keeps_good_items(predictor):
    results = predictor.predict_batch(["سٺو شاندار", "", None, "خراب ڏکيو"], errors="none")
    assert [r.label if r else None for r in results] == ["positive", None, None, "negative"]


# ------------------------------------------------------------- preprocessing

def test_text_is_cleaned_like_in_training(predictor):
    messy = "  سٺو\u200b   شاندار\u0640  "
    assert predictor.predict(messy).probabilities == predictor.predict("سٺو شاندار").probabilities
    assert predictor.predict(messy).text == messy  # the result echoes the input as given


def test_preprocessing_json_overrides_defaults(model_dir):
    assert SentimentPredictor.from_pretrained(model_dir).info.preprocessing_source.startswith("defaults")
    write_preprocessing_config(model_dir, CleaningOptions(char_map={"ڪ": "ك"}), source="test")
    pre = TextPreprocessor.from_model_dir(model_dir)
    assert pre.source == "preprocessing.json"
    assert pre.prepare("ڪتاب") == "كتاب"
    assert SentimentPredictor.from_pretrained(model_dir).info.preprocessing_source == "preprocessing.json"


def test_invalid_preprocessing_json_is_reported(model_dir):
    (model_dir / "preprocessing.json").write_text(json.dumps({"cleaning": {"bogus": 1}}), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="preprocessing.json"):
        SentimentPredictor.from_pretrained(model_dir)


# ---------------------------------------------------------------- model files

def test_missing_model_folder(tmp_path):
    with pytest.raises(ModelNotFoundError, match="not found"):
        SentimentPredictor.from_pretrained(tmp_path / "nope")


def test_folder_without_model_files(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ModelNotFoundError, match="No model files"):
        SentimentPredictor.from_pretrained(tmp_path / "empty")


def test_corrupt_model_file(tmp_path):
    directory = tmp_path / "bad"
    directory.mkdir()
    (directory / "model.joblib").write_bytes(b"this is not a model")
    with pytest.raises(ModelLoadError, match="Could not load"):
        SentimentPredictor.from_pretrained(directory)


# ------------------------------------------------------------------ metadata

def test_info_describes_the_model(predictor, model_dir):
    info = predictor.info
    assert isinstance(info, ModelInfo)
    assert info.name == "toy" and info.backend == "sklearn" and info.seed == 42
    assert info.labels == ("negative", "neutral", "positive") == predictor.labels
    assert info.confidence_type == "score"
    assert info.to_dict()["labels"] == ["negative", "neutral", "positive"]


def test_confidence_threshold_flags_uncertain_predictions(model_dir):
    strict = SentimentPredictor.from_pretrained(model_dir, confidence_threshold=1.0)
    assert strict.predict("سٺو شاندار").is_confident is False
    lenient = SentimentPredictor.from_pretrained(model_dir, confidence_threshold=0.0)
    assert lenient.predict("سٺو").is_confident is True
    with pytest.raises(ValueError):
        SentimentPredictor.from_pretrained(model_dir, confidence_threshold=1.5)


def test_from_config(model_dir, tmp_path):
    config = tmp_path / "inference.yaml"
    config.write_text(yaml.safe_dump({"model_dir": "models/toy", "batch_size": 4, "confidence_threshold": 0.4}),
                      encoding="utf-8")
    predictor = SentimentPredictor.from_config(config, project_root=tmp_path)
    assert predictor.predict("سٺو شاندار").label == "positive"


# --------------------------------------------- the transformers path, with a fake model

def _fake_transformers_predictor(logits_by_text):
    labels = ["negative", "neutral", "positive"]

    def score_fn(texts):
        return np.array([logits_by_text[t] for t in texts])

    backend = Backend("transformers", labels, score_fn, confidence_type="probability", device="cpu", max_length=128)
    info = ModelInfo("fake", "transformers", "x", tuple(labels), "probability", "cpu", 128)
    return SentimentPredictor(backend, TextPreprocessor(), info, batch_size=2)


def test_logits_become_probabilities():
    predictor = _fake_transformers_predictor({"a": [0.0, 0.0, 3.0], "b": [2.0, 0.0, 0.0], "c": [1000.0, 0.0, 0.0]})
    a, b, c = predictor.predict_batch(["a", "b", "c"])
    assert a.label == "positive" and b.label == "negative"
    assert a.probabilities["positive"] == pytest.approx(math.exp(3) / (math.exp(3) + 2))
    assert c.confidence == pytest.approx(1.0)  # huge logits do not overflow


def test_wrong_score_shape_is_reported():
    labels = ["a", "b"]
    backend = Backend("sklearn", labels, lambda texts: np.zeros((len(texts), 3)), "score")
    predictor = SentimentPredictor(backend, TextPreprocessor(), ModelInfo("m", "sklearn", "x", tuple(labels), "score"))
    with pytest.raises(ModelLoadError, match="shape"):
        predictor.predict("متن")


def test_softmax_is_stable_and_normalized():
    result = softmax(np.array([[1000.0, 1000.0, 1000.0], [0.0, 1.0, 2.0]]))
    assert np.allclose(result.sum(axis=1), 1.0) and np.isfinite(result).all()


# ------------------------------------------------------------- package exports

def test_lazy_package_exports():
    assert sindhi_nlp.SentimentPredictor is SentimentPredictor
    assert sindhi_nlp.InvalidInputError is InvalidInputError
    assert "SentimentPredictor" in dir(sindhi_nlp)
    with pytest.raises(AttributeError):
        sindhi_nlp.DoesNotExist