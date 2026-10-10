import json

import pandas as pd
import pytest
import yaml

from conftest import base_config
from sindhi_nlp.evaluation.compare_models import compare_models, exact_mcnemar, wilson_interval

N = 100


def _write_run(root, model, wrong_ids, truncate_to=None):
    metrics = root / "artifacts" / "metrics"
    reports = root / "artifacts" / "reports"
    metrics.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    accuracy = (N - len(wrong_ids)) / N
    (metrics / f"{model}_external_test.json").write_text(
        json.dumps({"metrics": {"num_examples": N, "accuracy": accuracy, "macro_f1": accuracy}}), encoding="utf-8")
    ids = wrong_ids if truncate_to is None else wrong_ids[:truncate_to]
    pd.DataFrame({"Text": [f"sentence {i}" for i in ids], "true_label": "negative", "predicted_label": "positive"}) \
        .to_csv(reports / f"{model}_external_test_misclassified.csv", index=False, encoding="utf-8")


@pytest.fixture
def root(tmp_path):
    base = base_config()
    base["paths"].update({"metrics_dir": "artifacts/metrics", "model_reports_dir": "artifacts/reports"})
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    return tmp_path


def test_exact_mcnemar_known_values():
    assert exact_mcnemar(10, 2) == pytest.approx(2 * (1 + 12 + 66) / 4096)
    assert exact_mcnemar(5, 5) == pytest.approx(1.0)
    assert exact_mcnemar(0, 0) == 1.0


def test_wilson_interval():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high and 0.40 < low < 0.41 and 0.59 < high < 0.60
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_compare_counts_agreement_and_pvalue(root):
    a_wrong = list(range(0, 20))            # model A wrong on 0..19
    b_wrong = list(range(10, 22))           # model B wrong on 10..21
    _write_run(root, "model_a", a_wrong)
    _write_run(root, "model_b", b_wrong)
    r = compare_models("model_a", "model_b", base_config_path=root / "base.yaml", project_root=root)
    assert r["agreement"] == {"both_wrong": 10, "only_a_wrong": 10, "only_b_wrong": 2, "both_right": 78}
    assert r["mcnemar_exact_p_value"] == pytest.approx(exact_mcnemar(10, 2))
    assert r["models"]["a"]["accuracy"] == 0.80 and r["models"]["b"]["accuracy"] == 0.88
    assert (root / "artifacts" / "metrics" / "compare_model_a_vs_model_b_external_test.json").is_file()


def test_truncated_error_list_is_rejected(root):
    _write_run(root, "model_a", list(range(20)), truncate_to=15)
    _write_run(root, "model_b", list(range(5)))
    with pytest.raises(ValueError, match="truncated"):
        compare_models("model_a", "model_b", base_config_path=root / "base.yaml", project_root=root)


def test_missing_scores_give_clear_error(root):
    _write_run(root, "model_a", [1, 2])
    with pytest.raises(FileNotFoundError, match="model_b"):
        compare_models("model_a", "model_b", base_config_path=root / "base.yaml", project_root=root)