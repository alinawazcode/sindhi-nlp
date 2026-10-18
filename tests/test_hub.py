"""Loading models by Hugging Face name. The Hub itself is never contacted: downloads are faked."""

import sys
import types

import joblib
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sindhi_nlp.cli.main import main
from sindhi_nlp.inference import hub
from sindhi_nlp.inference.predictor import SentimentPredictor
from sindhi_nlp.inference.schemas import ModelLoadError, ModelNotFoundError

REPO = "alinawazmahar/sindhi-sentiment-xlmr"
TRAIN = (
    [("خراب ڏکيو بڪار", "negative")] * 20 + [("شهر ڪتاب ميز", "neutral")] * 20 + [("سٺو شاندار خوش", "positive")] * 20
)


@pytest.fixture
def cached_model(tmp_path):
    """A model folder standing in for a downloaded Hub snapshot."""
    texts, labels = zip(*TRAIN)
    pipeline = Pipeline([("tfidf", TfidfVectorizer(token_pattern=r"\S+")), ("svc", LinearSVC(random_state=0))])
    pipeline.fit(list(texts), list(labels))
    folder = tmp_path / "hub_cache" / "snapshot"
    folder.mkdir(parents=True)
    joblib.dump(pipeline, folder / "model.joblib")
    return folder


@pytest.fixture
def fake_download(monkeypatch, cached_model):
    calls = []

    def download(repo_id, revision=None, cache_dir=None):
        calls.append({"repo_id": repo_id, "revision": revision, "cache_dir": cache_dir})
        return cached_model

    monkeypatch.setattr(hub, "download_model", download)
    return calls


@pytest.mark.parametrize(
    "name, expected",
    [
        (REPO, True),
        ("owner/name.v1", True),
        ("owner/my_model-2", True),
        ("artifacts/models/xlmr_mixed", False),
        ("single", False),
        ("./owner/name", False),
        ("/owner/name", False),
        ("C:\\owner\\name", False),
        ("owner\\name", False),
        ("", False),
    ],
)
def test_looks_like_repo_id(name, expected):
    assert hub.looks_like_repo_id(name) is expected


def test_loads_by_hub_name(fake_download):
    predictor = SentimentPredictor.from_pretrained(REPO, revision="v0.1.0", cache_dir="cache")
    assert predictor.predict("سٺو شاندار خوش").label == "positive"
    assert fake_download == [{"repo_id": REPO, "revision": "v0.1.0", "cache_dir": "cache"}]
    assert predictor.info.name == REPO  # not the cache folder's hash


def test_local_folder_wins_over_a_hub_name(fake_download, cached_model, tmp_path, monkeypatch):
    local = tmp_path / "alinawazmahar" / "sindhi-sentiment-xlmr"
    local.parent.mkdir()
    cached_model.rename(local)
    monkeypatch.chdir(tmp_path)
    SentimentPredictor.from_pretrained(REPO)
    assert fake_download == []  # no download: the local folder was used


def test_path_objects_never_reach_the_hub(fake_download, tmp_path):
    with pytest.raises(ModelNotFoundError, match="not found"):
        SentimentPredictor.from_pretrained(tmp_path / "alinawazmahar" / "sindhi-sentiment-xlmr")
    assert fake_download == []


def test_missing_local_path_is_reported_not_downloaded(fake_download):
    with pytest.raises(ModelNotFoundError, match="not found"):
        SentimentPredictor.from_pretrained("artifacts/models/does_not_exist")
    assert fake_download == []


def test_from_config_accepts_a_hub_name(fake_download, tmp_path):
    config = tmp_path / "inference.yaml"
    config.write_text(yaml.safe_dump({"model_dir": REPO, "revision": "main"}), encoding="utf-8")
    predictor = SentimentPredictor.from_config(config, project_root=tmp_path)
    assert predictor.predict("خراب ڏکيو").label == "negative"
    assert fake_download[0]["repo_id"] == REPO and fake_download[0]["revision"] == "main"


def test_cli_accepts_a_hub_name(fake_download, tmp_path, capsys):
    assert main(["predict", "سٺو شاندار", "--model-dir", REPO, "--project-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("positive")
    assert fake_download[0]["repo_id"] == REPO


# ---- download_model itself, with a fake huggingface_hub module ----


def test_download_model_requests_only_safe_files(monkeypatch, tmp_path):
    seen = {}

    def snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    folder = hub.download_model(REPO, revision="abc123", cache_dir=tmp_path / "c")
    assert folder == tmp_path
    assert seen["repo_id"] == REPO and seen["revision"] == "abc123"
    assert "*.bin" not in seen["allow_patterns"] and "*.safetensors" in seen["allow_patterns"]


def test_download_failure_is_a_clear_error(monkeypatch):
    def snapshot_download(**kwargs):
        raise OSError("no network")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    with pytest.raises(ModelNotFoundError, match="could not be downloaded.*no network"):
        hub.download_model(REPO)


def test_missing_huggingface_hub_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ModelLoadError, match="pip install huggingface_hub"):
        hub.download_model(REPO)


def test_local_models_keep_their_folder_name(cached_model):
    assert SentimentPredictor.from_pretrained(cached_model).info.name == "snapshot"


def test_cli_without_model_dir_or_config_uses_the_published_model(fake_download, tmp_path, capsys):
    assert main(["predict", "سٺو شاندار", "--project-root", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("positive")
    assert REPO in captured.err and "1.1 GB" in captured.err
    assert fake_download[0]["repo_id"] == REPO
