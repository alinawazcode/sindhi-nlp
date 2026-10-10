import json

import joblib
import pandas as pd
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from conftest import LABELS, base_config
from sindhi_nlp.data.prepare_external import run_external_prep
from sindhi_nlp.evaluation.evaluate_external import FinalEvaluationError, evaluate_model

CAPITALIZED = {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"}


def _w(i: int) -> str:
    """A letters-only word unique to ``i`` (digits would be ignored by the near-duplicate key)."""
    return "".join(chr(0x0628 + int(d)) for d in str(i))


def _natural_rows():
    rows = [(f"جملو {label} {_w(i)}", CAPITALIZED[label]) for label in LABELS for i in range(100)]
    rows += [(f"جملو negative {_w(i)}", "Negative") for i in range(5)]            # exact duplicates
    rows += [(f"جملو positive {_w(i)}!", "Positive") for i in range(5)]           # punctuation variants
    rows += [("   ", "Positive"), ("متن", "Mixed")]                            # blank text, invalid label
    return rows


@pytest.fixture
def ext_project(tmp_path):
    root = tmp_path
    (root / "data" / "external").mkdir(parents=True)
    pd.DataFrame(_natural_rows(), columns=["Sindhi Text", "Sentiment"]).to_csv(
        root / "data" / "external" / "natural_sindhi.csv", index=False, encoding="utf-8"
    )
    # Synthetic splits that happen to contain 10 natural texts (contamination).
    leaked = [f"جملو negative {_w(i)}" for i in range(50, 55)] + [f"جملو positive {_w(i)}" for i in range(50, 55)]
    (root / "data" / "processed").mkdir(parents=True)
    pd.DataFrame({"Text": leaked + ["synthetic only"], "Label": ["negative"] * 11}).to_csv(
        root / "data" / "processed" / "train.csv", index=False, encoding="utf-8"
    )
    base = root / "base.yaml"
    base.write_text(yaml.safe_dump(base_config(), allow_unicode=True), encoding="utf-8")
    ext = root / "external.yaml"
    ext.write_text(
        yaml.safe_dump(
            {
                "source_file": "data/external/natural_sindhi.csv",
                "text_column": "Sindhi Text",
                "label_column": "Sentiment",
                "duplicate_policy": "drop",
                "split": {"train": 0.7, "validation": 0.1, "test": 0.2},
                "output_dir": "data/external/processed",
                "compare_with": ["data/processed/train.csv", "data/processed/validation.csv"],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return root, base, ext


def test_prepare_external_end_to_end(ext_project):
    root, base, ext = ext_project
    report = run_external_prep(base, ext, root)

    assert report["source"]["rows"] == 312
    assert report["validation"]["missing_text"] == 1
    assert report["validation"]["invalid_label_rows"] == 1
    assert report["duplicates"]["removed_duplicate_rows"] == 5
    assert report["total_rows_after_cleaning"] == 305
    assert sum(s["rows"] for s in report["splits"].values()) == 305

    # No text or near-duplicate group crosses a split boundary.
    zero = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    assert report["cross_split_text_overlap"] == zero
    assert report["cross_split_group_overlap"] == zero

    assert report["synthetic_overlap"]["total_rows"] == 10
    assert report["synthetic_overlap"]["compared_files"] == ["train.csv"]

    out = root / "data" / "external" / "processed"
    for name in ("train", "validation", "test"):
        frame = pd.read_csv(out / f"{name}.csv")
        assert list(frame.columns) == ["Text", "Label"]
        assert set(frame["Label"]) <= set(LABELS)  # capitalized labels were normalized
    assert (out / "split_report.json").is_file()


def test_prepare_external_is_reproducible(ext_project, tmp_path_factory):
    root, base, ext = ext_project
    a = run_external_prep(base, ext, root)
    b = run_external_prep(base, ext, root)
    assert a["outputs_sha256"] == b["outputs_sha256"]


def test_prepare_external_refuses_raw_directory(ext_project):
    root, base, ext = ext_project
    cfg = yaml.safe_load(ext.read_text(encoding="utf-8"))
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "external" / "natural_sindhi.csv").replace(root / "data" / "raw" / "natural_sindhi.csv")
    cfg["source_file"] = "data/raw/natural_sindhi.csv"
    ext.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="out of data/raw"):
        run_external_prep(base, ext, root)


def test_prepare_external_missing_file(ext_project):
    root, base, ext = ext_project
    (root / "data" / "external" / "natural_sindhi.csv").unlink()
    with pytest.raises(FileNotFoundError):
        run_external_prep(base, ext, root)


@pytest.fixture
def evaluated_project(ext_project):
    root, base, ext = ext_project
    run_external_prep(base, ext, root)
    train = pd.read_csv(root / "data" / "external" / "processed" / "train.csv")
    model = Pipeline([("tfidf", TfidfVectorizer(token_pattern=r"\S+")), ("svc", LinearSVC(random_state=0))])
    model.fit(train["Text"], train["Label"])
    model_dir = root / "artifacts" / "models" / "toy_model"
    model_dir.mkdir(parents=True)
    joblib.dump(model, model_dir / "model.joblib")
    return root, base, model_dir


def test_evaluate_validation_split(evaluated_project):
    root, base, model_dir = evaluated_project
    report = evaluate_model(model_dir, "data/external/processed/validation.csv", "validation",
                            base_config_path=base, project_root=root)
    assert report["dataset"]["split"] == "validation"
    assert 0.0 <= report["metrics"]["macro_f1"] <= 1.0
    assert (root / "artifacts" / "metrics" / "toy_model_external_validation.json").is_file()
    assert (root / "artifacts" / "reports" / "toy_model_external_validation_misclassified.csv").is_file()
    # Validation can be re-scored freely.
    evaluate_model(model_dir, "data/external/processed/validation.csv", "validation",
                   base_config_path=base, project_root=root)


def test_test_split_needs_final_flag_and_runs_once(evaluated_project):
    root, base, model_dir = evaluated_project
    kwargs = dict(base_config_path=base, project_root=root)
    with pytest.raises(FinalEvaluationError, match="--final"):
        evaluate_model(model_dir, "data/external/processed/test.csv", "test", **kwargs)
    # The same guard applies when the test file is passed under another split name.
    with pytest.raises(FinalEvaluationError, match="--final"):
        evaluate_model(model_dir, "data/external/processed/test.csv", "validation", **kwargs)

    evaluate_model(model_dir, "data/external/processed/test.csv", "test", final=True, **kwargs)
    result = root / "artifacts" / "metrics" / "toy_model_external_test.json"
    assert json.loads(result.read_text("utf-8"))["dataset"]["split"] == "test"

    with pytest.raises(FinalEvaluationError, match="already"):
        evaluate_model(model_dir, "data/external/processed/test.csv", "test", final=True, **kwargs)
    evaluate_model(model_dir, "data/external/processed/test.csv", "test", final=True, overwrite=True, **kwargs)


def test_evaluate_requires_a_model(ext_project):
    root, base, ext = ext_project
    run_external_prep(base, ext, root)
    with pytest.raises(FileNotFoundError, match="model.joblib"):
        evaluate_model("artifacts/models/missing", "data/external/processed/validation.csv", "validation",
                       base_config_path=base, project_root=root)