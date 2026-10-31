"""The shipped YAML configs must load and validate, so a broken edit is caught by CI."""

from pathlib import Path

import pytest

from sindhi_nlp.training.trainer import load_training_config
from sindhi_nlp.utils.config import load_base_config, load_yaml

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_base_config_is_valid():
    assert load_base_config(CONFIGS / "base.yaml")["seed"] == 42


@pytest.mark.parametrize("name", ["xlm_roberta_base.yaml", "xlmr_mixed.yaml"])
def test_training_configs_are_valid(name):
    cfg = load_training_config(CONFIGS / name)
    assert cfg["model_name"] == "xlm-roberta-base"


def test_published_model_config_matches_its_training_run():
    cfg = load_training_config(CONFIGS / "xlmr_mixed.yaml")
    assert cfg["run_name"] == "xlmr_mixed"
    assert cfg["training"]["num_train_epochs"] == 3
    assert cfg["max_rows"] == {"synthetic": 30000}
    assert cfg["repeat"] == {"natural": 5}


@pytest.mark.parametrize("name", ["inference.yaml", "baseline.yaml", "external.yaml"])
def test_other_configs_parse(name):
    assert isinstance(load_yaml(CONFIGS / name), dict)
