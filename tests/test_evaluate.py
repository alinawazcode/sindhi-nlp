"""Tests for error analysis, model evaluation, plots, ``--version`` and ``python -m sindhi_nlp``."""

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import sindhi_nlp
from sindhi_nlp.evaluation import evaluate as ev
from sindhi_nlp.evaluation.error_analysis import build_results_frame, confusion_pairs, misclassified
from sindhi_nlp.inference.predictor import SentimentPredictor

TRAIN = (
    [("خراب ڏکيو بڪار", "negative")] * 20 + [("شهر ڪتاب ميز", "neutral")] * 20 + [("سٺو شاندار خوش", "positive")] * 20
)


@pytest.fixture
def model_dir(tmp_path):
    texts, labels = zip(*TRAIN)
    pipeline = Pipeline([("tfidf", TfidfVectorizer(token_pattern=r"\S+")), ("svc", LinearSVC(random_state=0))])
    pipeline.fit(list(texts), list(labels))
    directory = tmp_path / "toy"
    directory.mkdir()
    joblib.dump(pipeline, directory / "model.joblib")
    (directory / "config.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
    return directory


@pytest.fixture
def predictor(model_dir):
    return SentimentPredictor.from_pretrained(model_dir)


# ---------------------------------------------------------------- error analysis


def test_results_frame_and_misclassified_order():
    frame = build_results_frame(
        ["a", "b", "c"], ["positive", "negative", "neutral"], ["positive", "positive", "negative"], [0.9, 0.6, 0.8]
    )
    assert list(frame.columns) == ["text", "true", "predicted", "confidence", "correct"]
    assert frame["correct"].tolist() == [True, False, False]
    wrong = misclassified(frame)
    assert wrong["text"].tolist() == ["c", "b"]  # most confident mistake first
    assert len(misclassified(frame, limit=1)) == 1


def test_confusion_pairs_counts():
    frame = build_results_frame(["a", "b", "c"], ["negative"] * 3, ["positive", "positive", "neutral"])
    pairs = confusion_pairs(frame)
    top = pairs.iloc[0]
    assert (top["true"], top["predicted"], top["count"]) == ("negative", "positive", 2)


def test_results_frame_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_results_frame(["a"], ["positive", "negative"], ["positive"])


# --------------------------------------------------------------------- evaluate


def test_evaluate_predictor_scores_and_skips(predictor):
    texts = ["سٺو شاندار خوش", "خراب ڏکيو بڪار", "   ", "شهر ڪتاب ميز"]
    gold = ["positive", "negative", "neutral", "neutral"]
    metrics, results = ev.evaluate_predictor(predictor, texts, gold)
    assert metrics["skipped"] == 1
    assert len(results) == 3
    assert metrics["accuracy"] == 1.0
    assert "macro" in ev.format_report(metrics, "toy").lower() or "f1" in ev.format_report(metrics).lower()


def test_evaluate_predictor_validates_inputs(predictor):
    with pytest.raises(ValueError, match="differ in length"):
        ev.evaluate_predictor(predictor, ["a"], [])
    with pytest.raises(ValueError, match="not labels"):
        ev.evaluate_predictor(predictor, ["سٺو"], ["happy"])
    with pytest.raises(ValueError, match="None of the texts"):
        ev.evaluate_predictor(predictor, ["  "], ["positive"])


def test_evaluate_cli_writes_outputs(model_dir, tmp_path, capsys):
    csv = tmp_path / "gold.csv"
    pd.DataFrame({"Text": ["سٺو شاندار خوش", "خراب ڏکيو بڪار"], "Label": [" Positive", "NEGATIVE "]}).to_csv(
        csv, index=False, encoding="utf-8-sig"
    )
    out = tmp_path / "out" / "metrics.json"
    code = ev.main(["--model-dir", str(model_dir), "--csv", str(csv), "--output", str(out), "--show-errors", "2"])
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["accuracy"] == 1.0
    assert (tmp_path / "out" / "metrics_misclassified.csv").exists()
    assert "accuracy" in capsys.readouterr().out.lower()


def test_evaluate_cli_reports_bad_column(model_dir, tmp_path, capsys):
    csv = tmp_path / "gold.csv"
    pd.DataFrame({"a": ["x"], "b": ["positive"]}).to_csv(csv, index=False)
    assert ev.main(["--model-dir", str(model_dir), "--csv", str(csv)]) == 1
    assert "Column 'Text' not found" in capsys.readouterr().err


# ------------------------------------------------------------------------ plots


def test_plots_write_png(tmp_path):
    pytest.importorskip("matplotlib")
    from sindhi_nlp.evaluation import plots

    labels = ["negative", "neutral", "positive"]
    matrix = [[8, 1, 1], [0, 9, 1], [1, 0, 9]]
    for normalize in (False, True):
        path = tmp_path / f"cm{normalize}.png"
        plots.plot_confusion_matrix(matrix, labels, path=path, normalize=normalize)
        assert path.read_bytes()[:4] == b"\x89PNG"
    plots.plot_model_comparison({"a": 0.8, "b": 0.9}, tmp_path / "cmp.png")
    plots.plot_per_label_f1(
        {"negative": {"f1": 0.9}, "neutral": {"f1": 0.8}, "positive": {"f1": 0.85}}, tmp_path / "f1.png"
    )
    assert (tmp_path / "cmp.png").exists() and (tmp_path / "f1.png").exists()


# --------------------------------------------------------------------- version


def test_version_flag_and_module_entry():
    src = str(Path(sindhi_nlp.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src}
    result = subprocess.run(
        [sys.executable, "-m", "sindhi_nlp", "--version"], capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"sindhi-nlp {sindhi_nlp.__version__}"
