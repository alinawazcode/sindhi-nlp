import json

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from conftest import base_config
from sindhi_nlp.cli.main import main
from sindhi_nlp.inference.predictor import SentimentPredictor

TRAIN = (
    [("خراب ڏکيو بڪار", "negative")] * 20
    + [("شهر ڪتاب ميز", "neutral")] * 20
    + [("سٺو شاندار خوش", "positive")] * 20
)


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
        yaml.safe_dump({"model_dir": "artifacts/models/toy", "device": "cpu"}), encoding="utf-8")
    (tmp_path / "configs" / "base.yaml").write_text(yaml.safe_dump(base_config()), encoding="utf-8")
    return tmp_path


def run(project, *args):
    return main([*args, "--project-root", str(project)] if args[0] in ("predict", "batch", "info") else list(args))


def test_predict_table(project, capsys):
    assert run(project, "predict", "سٺو شاندار", "خراب ڏکيو") == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("positive") and "سٺو شاندار" in lines[0]
    assert lines[1].startswith("negative")


def test_predict_json_and_invalid_text(project, capsys):
    assert run(project, "predict", "سٺو", "   ", "--json") == 1  # one text is blank
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["label"] == "positive" and payload[1] is None


def test_model_dir_option_overrides_config(project, capsys):
    assert run(project, "predict", "سٺو", "--model-dir", "artifacts/models/toy") == 0
    assert capsys.readouterr().out.startswith("positive")


def test_info(project, capsys):
    assert run(project, "info", "--json") == 0
    info = json.loads(capsys.readouterr().out)
    assert info["backend"] == "sklearn" and info["labels"] == ["negative", "neutral", "positive"]


def test_batch_csv_with_a_bad_row(project, capsys):
    pd.DataFrame({"Text": ["سٺو شاندار", "", "خراب ڏکيو"], "id": ["1", "2", "3"]}).to_csv(
        project / "in.csv", index=False, encoding="utf-8")
    assert run(project, "batch", "--input", "in.csv", "--output", "out/results.csv") == 0
    out = pd.read_csv(project / "out" / "results.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(out["predicted_label"]) == ["positive", "", "negative"]
    assert list(out["error"]) == ["", "invalid input", ""]
    assert {"id", "confidence", "prob_negative", "prob_neutral", "prob_positive"} <= set(out.columns)
    assert "Classified 2 of 3 rows" in capsys.readouterr().out  # the blank row is not counted


def test_batch_text_file_and_custom_column(project):
    (project / "lines.txt").write_text("سٺو شاندار\nخراب ڏکيو\n", encoding="utf-8")
    assert run(project, "batch", "--input", "lines.txt", "--output", "lines_out.csv") == 0
    out = pd.read_csv(project / "lines_out.csv", dtype=str, encoding="utf-8-sig")
    assert list(out["predicted_label"]) == ["positive", "negative"]

    pd.DataFrame({"Sindhi Text": ["سٺو"]}).to_csv(project / "other.csv", index=False, encoding="utf-8")
    assert run(project, "batch", "--input", "other.csv", "--output", "o2.csv", "--text-column", "sindhi text") == 0


def test_errors_are_reported_not_raised(project, capsys):
    assert main(["predict", "x", "--model-dir", "nope", "--project-root", str(project)]) == 1
    assert "Model folder not found" in capsys.readouterr().err
    assert main(["batch", "--input", "missing.csv", "--output", "o.csv", "--project-root", str(project)]) == 1
    assert "not found" in capsys.readouterr().err
    pd.DataFrame({"x": ["a"]}).to_csv(project / "bad.csv", index=False)
    assert main(["batch", "--input", "bad.csv", "--output", "o.csv", "--project-root", str(project)]) == 1
    assert "Column 'Text' not found" in capsys.readouterr().err


def test_finalize_model_writes_preprocessing_json(project, capsys):
    model_dir = project / "artifacts" / "models" / "toy"
    assert main(["finalize-model", "--model-dir", str(model_dir), "--base-config", str(project / "configs" / "base.yaml")]) == 0
    saved = json.loads((model_dir / "preprocessing.json").read_text("utf-8"))
    assert saved["cleaning"]["unicode_form"] == "NFKC" and saved["cleaning"]["remove_tatweel"] is True
    assert SentimentPredictor.from_pretrained(model_dir).info.preprocessing_source == "preprocessing.json"
    assert main(["finalize-model", "--model-dir", str(project / "missing")]) == 1