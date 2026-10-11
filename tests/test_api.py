"""Tests for the HTTP API. Skipped automatically if fastapi or httpx is not installed."""

import joblib
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from sindhi_nlp import __version__  # noqa: E402
from sindhi_nlp.api.dependencies import Settings, load_settings  # noqa: E402
from sindhi_nlp.api.main import create_app  # noqa: E402
from sindhi_nlp.inference.predictor import SentimentPredictor  # noqa: E402
from sindhi_nlp.utils.config import ConfigError  # noqa: E402

TRAIN = (
    [("خراب ڏکيو بڪار", "negative")] * 20 + [("شهر ڪتاب ميز", "neutral")] * 20 + [("سٺو شاندار خوش", "positive")] * 20
)
POSITIVE, NEGATIVE = "سٺو شاندار خوش", "خراب ڏکيو بڪار"


@pytest.fixture
def project(tmp_path):
    texts, labels = zip(*TRAIN)
    pipeline = Pipeline([("tfidf", TfidfVectorizer(token_pattern=r"\S+")), ("svc", LinearSVC(random_state=0))])
    pipeline.fit(list(texts), list(labels))
    model_dir = tmp_path / "artifacts" / "models" / "toy"
    model_dir.mkdir(parents=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "inference.yaml").write_text(
        yaml.safe_dump({"model_dir": "artifacts/models/toy", "api": {"max_batch_size": 3}}), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def client(project):
    settings = load_settings(project_root=project)
    app = create_app(settings=settings)  # loads the model from the config at startup
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ------------------------------------------------------------------ status


def test_health_when_model_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model"] == "toy" and body["backend"] == "sklearn" and body["package_version"] == __version__


def test_model_info(client):
    body = client.get("/model").json()
    assert body["name"] == "toy" and body["labels"] == ["negative", "neutral", "positive"]
    assert body["confidence_type"] == "score"


def test_openapi_docs_are_served(client):
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


# --------------------------------------------------------------- predictions


def test_predict(client):
    response = client.post("/predict", json={"text": POSITIVE})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive" and body["model"] == "toy"
    assert body["text"] == POSITIVE  # Sindhi text survives the round trip unchanged
    assert set(body["probabilities"]) == {"negative", "neutral", "positive"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-9
    assert body["confidence"] == max(body["probabilities"].values()) and body["is_confident"] is True
    assert POSITIVE in response.content.decode("utf-8")  # sent as real UTF-8 text, not \\u escapes


def test_predict_cleans_text_like_training(client):
    messy = client.post("/predict", json={"text": "  سٺو\u200b  شاندار خوش\u0640 "}).json()
    clean = client.post("/predict", json={"text": POSITIVE}).json()
    assert messy["probabilities"] == clean["probabilities"]


def test_predict_batch_keeps_order(client):
    response = client.post("/predict/batch", json={"texts": [POSITIVE, NEGATIVE, "شهر ڪتاب"]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3 and body["model"] == "toy"
    assert [r["label"] for r in body["results"]] == ["positive", "negative", "neutral"]


# ----------------------------------------------------------- invalid requests


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and body["error"]["message"]
    return body["error"]


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": 5}, {"text": None}, {"txt": POSITIVE}, [POSITIVE]])
def test_predict_rejects_bad_payloads(client, payload):
    error = assert_error(client.post("/predict", json=payload), 422, "validation_error")
    assert error["details"] and all({"field", "message"} <= set(d) for d in error["details"])


def test_validation_errors_do_not_echo_the_input(client):
    secret = "x" * 10_001
    response = client.post("/predict", json={"text": secret})
    assert_error(response, 422, "validation_error")
    assert secret not in response.text


def test_predict_whitespace_only_is_invalid_input(client):
    error = assert_error(client.post("/predict", json={"text": "   \u200b "}), 422, "invalid_input")
    assert "empty" in error["message"] and "Item" not in error["message"]


def test_malformed_json(client):
    response = client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    assert_error(response, 422, "validation_error")


@pytest.mark.parametrize("payload", [{"texts": []}, {"texts": "one string"}, {"texts": [POSITIVE, ""]}, {}])
def test_batch_rejects_bad_payloads(client, payload):
    assert_error(client.post("/predict/batch", json=payload), 422, "validation_error")


def test_batch_names_the_bad_item(client):
    error = assert_error(
        client.post("/predict/batch", json={"texts": [POSITIVE, "  ", NEGATIVE]}), 422, "invalid_input"
    )
    assert "Item 1" in error["message"]


def test_batch_size_limit(client):
    error = assert_error(client.post("/predict/batch", json={"texts": [POSITIVE] * 4}), 422, "batch_too_large")
    assert "limit is 3" in error["message"]
    assert client.post("/predict/batch", json={"texts": [POSITIVE] * 3}).status_code == 200


def test_unknown_route_and_wrong_method(client):
    assert_error(client.get("/nope"), 404, "not_found")
    assert_error(client.get("/predict"), 405, "method_not_allowed")


# ------------------------------------------------------------ failure modes


def test_service_stays_up_when_the_model_cannot_load(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "inference.yaml").write_text(yaml.safe_dump({"model_dir": "missing"}), encoding="utf-8")
    app = create_app(settings=load_settings(project_root=tmp_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "unavailable" and health.json()["model_loaded"] is False
        assert "tmp" not in health.json()["detail"]  # no server paths leak to clients
        assert_error(client.post("/predict", json={"text": POSITIVE}), 503, "model_unavailable")
        assert_error(client.get("/model"), 503, "model_unavailable")


def test_unexpected_errors_are_hidden_from_clients(project):
    class Broken:
        info = SentimentPredictor.from_pretrained(project / "artifacts" / "models" / "toy").info

        def predict(self, text):
            raise RuntimeError("secret internal detail")

        def predict_batch(self, texts, errors="raise"):
            raise RuntimeError("secret internal detail")

    app = create_app(predictor=Broken(), settings=load_settings(project_root=project))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json={"text": POSITIVE})
        assert_error(response, 500, "internal_error")
        assert "secret" not in response.text


def test_predictor_can_be_injected(project):
    predictor = SentimentPredictor.from_pretrained(project / "artifacts" / "models" / "toy")
    app = create_app(predictor=predictor, settings=Settings(project / "none.yaml", project))
    with TestClient(app) as client:
        assert client.post("/predict", json={"text": NEGATIVE}).json()["label"] == "negative"


# ------------------------------------------------------------- settings / CORS


def test_cors_is_off_by_default_and_configurable(project):
    origin = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"}
    with TestClient(create_app(settings=load_settings(project_root=project))) as plain:
        assert "access-control-allow-origin" not in plain.options("/predict", headers=origin).headers

    settings = Settings(project / "configs" / "inference.yaml", project, 3, ("http://localhost:3000",))
    with TestClient(create_app(settings=settings)) as cors:
        assert (
            cors.options("/predict", headers=origin).headers["access-control-allow-origin"] == "http://localhost:3000"
        )


def test_settings_from_config_and_environment(project, monkeypatch):
    assert load_settings(project_root=project).max_batch_size == 3
    other = project / "other.yaml"
    other.write_text(
        yaml.safe_dump({"model_dir": "x", "api": {"max_batch_size": 9, "cors_origins": ["http://a"]}}), encoding="utf-8"
    )
    monkeypatch.setenv("SINDHI_NLP_CONFIG", str(other))
    monkeypatch.setenv("SINDHI_NLP_ROOT", str(project))
    settings = load_settings()
    assert settings.max_batch_size == 9 and settings.cors_origins == ("http://a",)
    assert settings.project_root == project.resolve()


@pytest.mark.parametrize(
    "api", [{"max_batch_size": 0}, {"max_batch_size": "ten"}, {"cors_origins": "http://a"}, "text"]
)
def test_invalid_api_settings_are_rejected(project, api):
    (project / "configs" / "inference.yaml").write_text(
        yaml.safe_dump({"model_dir": "x", "api": api}), encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_settings(project_root=project)


def test_missing_config_gives_default_settings(tmp_path):
    settings = load_settings(project_root=tmp_path)
    assert settings.max_batch_size == 64 and settings.cors_origins == ()
