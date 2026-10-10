import json
import random

import joblib
import pandas as pd
import pytest
import yaml

from conftest import LABELS, base_config
from sindhi_nlp.training import baseline
from sindhi_nlp.training.baseline import run_baseline

VOCAB = {
    "negative": ["ڏکيو", "خراب", "بڪار", "مايوس"],
    "neutral": ["اڄ", "شهر", "ڪتاب", "ميز"],
    "positive": ["سٺو", "شاندار", "خوش", "بهترين"],
}
NOISE = ["۽", "ته", "هي", "اهو", "پوءِ", "جڏهن"]


def _make_split(n_per_label: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        for i in range(n_per_label):
            words = rng.sample(VOCAB[label], 2) + rng.sample(NOISE, 3) + [f"w{rng.randint(0, 9999)}"]
            rng.shuffle(words)
            rows.append((" ".join(words), label))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["Text", "Label"])


def _baseline_yaml():
    return {
        "model_name": "baseline_test",
        "tfidf": {
            "analyzer": "word",
            "token_pattern": r"(?u)[\w\u064B-\u065F\u0670]+",
            "ngram_range": [1, 2],
            "min_df": 1,
            "sublinear_tf": True,
        },
        "svc": {"C": 1.0, "max_iter": 5000},
        "analysis": {"top_features_per_class": 5, "max_error_examples": 10},
    }


@pytest.fixture
def project(tmp_path):
    """A project with train and validation splits only. There is no test.csv."""
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    _make_split(150, seed=1).to_csv(processed / "train.csv", index=False, encoding="utf-8")
    _make_split(30, seed=2).to_csv(processed / "validation.csv", index=False, encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(base_config(), allow_unicode=True), encoding="utf-8")
    cfg = tmp_path / "baseline.yaml"
    cfg.write_text(yaml.safe_dump(_baseline_yaml(), allow_unicode=True), encoding="utf-8")
    return tmp_path, base, cfg


def test_baseline_learns_and_never_needs_test_split(project):
    root, base, cfg = project
    assert not (root / "data" / "processed" / "test.csv").exists()
    report = run_baseline(base, cfg, root)  # would raise if it tried to read test.csv
    assert report["validation_metrics"]["macro_f1"] > 0.9
    assert report["validation_metrics"]["num_examples"] == 90


def test_artifacts_saved_together(project):
    root, base, cfg = project
    run_baseline(base, cfg, root)
    model_dir = root / "artifacts" / "models" / "baseline_test"
    for name in ("model.joblib", "label_mapping.json", "config.json", "metrics.json"):
        assert (model_dir / name).is_file(), name
    assert json.loads((model_dir / "config.json").read_text("utf-8"))["seed"] == 42
    assert json.loads((model_dir / "label_mapping.json").read_text("utf-8")) == {
        "negative": 0, "neutral": 1, "positive": 2,
    }
    assert (root / "artifacts" / "metrics" / "baseline_test_validation.json").is_file()
    assert (root / "artifacts" / "reports" / "baseline_test_summary.md").is_file()
    assert (root / "artifacts" / "reports" / "baseline_test_misclassified_validation.csv").is_file()

    model = joblib.load(model_dir / "model.joblib")
    assert model.predict(["سٺو شاندار خوش"])[0] == "positive"


def test_results_are_reproducible(project):
    root, base, cfg = project
    a = run_baseline(base, cfg, root)
    b = run_baseline(base, cfg, root)
    assert a["validation_metrics"] == b["validation_metrics"]
    assert a["top_features"] == b["top_features"]
    assert a["data"] == b["data"]


def test_error_analysis_outputs(project):
    root, base, cfg = project
    report = run_baseline(base, cfg, root)
    errors = pd.read_csv(root / "artifacts" / "reports" / "baseline_test_misclassified_validation.csv")
    assert len(errors) <= 10
    assert (errors["true_label"] != errors["predicted_label"]).all()
    assert set(report["top_features"]) == set(LABELS)
    assert all(len(t) == 5 for t in report["top_features"].values())


def test_unknown_label_in_split_is_rejected(project):
    root, base, cfg = project
    path = root / "data" / "processed" / "validation.csv"
    df = pd.read_csv(path)
    df.loc[0, "Label"] = "mixed"
    df.to_csv(path, index=False, encoding="utf-8")
    with pytest.raises(ValueError, match="mixed"):
        run_baseline(base, cfg, root)


def test_source_never_references_test_split():
    import inspect

    source = inspect.getsource(baseline)
    assert "test.csv" not in source.replace("``test.csv``", "")


def test_baseline_on_another_data_dir_with_its_own_name(project):
    import shutil

    root, base, cfg = project
    other = root / "data" / "natural"
    shutil.copytree(root / "data" / "processed", other)
    first = run_baseline(base, cfg, root)
    second = run_baseline(base, cfg, root, data_dir="data/natural", model_name="baseline_natural")

    assert second["model_name"] == "baseline_natural" and second["data_dir"] == "data/natural"
    assert (root / "artifacts" / "models" / "baseline_natural" / "model.joblib").is_file()
    assert (root / "artifacts" / "models" / "baseline_test" / "model.joblib").is_file()  # first run untouched
    assert (root / "artifacts" / "reports" / "baseline_natural_summary.md").is_file()
    assert first["validation_metrics"] == second["validation_metrics"]  # same data, same result