"""Tests for the parts of Phase 4 that do not need torch or transformers."""
import json
import types

import pandas as pd
import pytest
import yaml

from conftest import LABELS, base_config
from sindhi_nlp.training.callbacks import JsonlLogCallback
from sindhi_nlp.training.trainer import (
    EncodedDataset,
    build_datasets,
    build_training_kwargs,
    encode_dataframe,
    load_split,
    load_training_config,
    stratified_sample,
    token_length_stats,
    validate_training_config,
)
from sindhi_nlp.utils.config import ConfigError, load_base_config

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


def _w(i: int) -> str:
    return "".join(chr(0x0628 + int(d)) for d in str(i))


def _frame(prefix: str, n_per_label: int, start: int = 0) -> pd.DataFrame:
    rows = [(f"{prefix} {label} {_w(i)}", label) for label in LABELS for i in range(start, start + n_per_label)]
    return pd.DataFrame(rows, columns=["Text", "Label"])


def training_config(**overrides) -> dict:
    cfg = {
        "model_name": "xlm-roberta-base",
        "run_name": "test_run",
        "datasets": {"synthetic": "data/processed", "natural": "data/external/processed"},
        "train_on": ["synthetic", "natural"],
        "validate_on": ["natural"],
        "report_on": ["synthetic"],
        "max_rows": {},
        "repeat": {},
        "tokenizer": {"max_length": 32},
        "training": {
            "learning_rate": 2.0e-5, "per_device_train_batch_size": 8, "per_device_eval_batch_size": 16,
            "num_train_epochs": 3, "warmup_ratio": 0.1, "weight_decay": 0.01, "early_stopping_patience": 2,
        },
        "smoke": {"train_rows": 60, "validation_rows": 30, "epochs": 1, "batch_size": 4},
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def project(tmp_path):
    """Synthetic and natural splits. There is deliberately no test.csv anywhere."""
    for folder, prefix in (("data/processed", "syn"), ("data/external/processed", "nat")):
        directory = tmp_path / folder
        directory.mkdir(parents=True)
        _frame(prefix, 40, 0).to_csv(directory / "train.csv", index=False, encoding="utf-8")
        _frame(prefix, 10, 100).to_csv(directory / "validation.csv", index=False, encoding="utf-8")
    # A natural validation text that also appears in the synthetic training data (contamination).
    leaked = pd.read_csv(tmp_path / "data/external/processed/validation.csv").iloc[[0]]
    syn_train = pd.concat([pd.read_csv(tmp_path / "data/processed/train.csv"), leaked], ignore_index=True)
    syn_train.to_csv(tmp_path / "data/processed/train.csv", index=False, encoding="utf-8")
    base = base_config()
    base["paths"].update({"models_dir": "artifacts/models", "metrics_dir": "artifacts/metrics",
                          "model_reports_dir": "artifacts/reports"})
    return tmp_path, base, leaked["Text"].iloc[0]


def test_test_split_can_never_be_loaded(project):
    root, base, _ = project
    with pytest.raises(ValueError, match="test split"):
        load_split(training_config(), root, "natural", "test", LABELS, "Text", "Label")


def test_build_datasets_combines_dedupes_and_protects_validation(project):
    root, base, leaked_text = project
    data = build_datasets(training_config(), base, root, seed=42)
    assert leaked_text not in set(data.train["Text"])
    assert data.info["removed_validation_overlap"] == 1
    assert data.info["train_rows"] == 120 + 120 + 1 - 1  # synthetic + natural, minus the contaminated row
    assert data.info["validation_rows"] == 30 and data.info["report_rows"] == {"synthetic": 30}
    assert set(data.validation["Text"]).isdisjoint(data.train["Text"])
    assert set(data.report["synthetic"]["Text"]).isdisjoint(data.train["Text"])
    assert set(data.validation["Label"]) == set(LABELS)
    assert sorted(data.file_hashes) == [
        "natural/train.csv", "natural/validation.csv", "synthetic/train.csv", "synthetic/validation.csv"
    ]


def test_build_datasets_is_deterministic(project):
    root, base, _ = project
    a = build_datasets(training_config(), base, root, seed=42)
    b = build_datasets(training_config(), base, root, seed=42)
    c = build_datasets(training_config(), base, root, seed=7)
    pd.testing.assert_frame_equal(a.train, b.train)
    assert not a.train.equals(c.train)


def test_max_rows_and_repeat(project):
    root, base, _ = project
    cfg = training_config(max_rows={"synthetic": 30}, repeat={"natural": 3})
    data = build_datasets(cfg, base, root, seed=42)
    natural_rows = data.train["Text"].str.startswith("nat").sum()
    synthetic_rows = data.train["Text"].str.startswith("syn").sum()
    assert synthetic_rows == 30 and natural_rows == 3 * 120


def test_smoke_mode_shrinks_everything(project):
    root, base, _ = project
    data = build_datasets(training_config(), base, root, seed=42, smoke=True)
    assert len(data.train) == 60 and len(data.validation) == 30
    assert all(len(df) == 30 for df in data.report.values())
    assert set(data.train["Label"]) == set(LABELS)


def test_missing_report_dataset_is_skipped_but_missing_training_data_fails(project):
    root, base, _ = project
    (root / "data/processed/validation.csv").unlink()
    data = build_datasets(training_config(), base, root, seed=42)
    assert data.report == {}
    (root / "data/external/processed/train.csv").unlink()
    with pytest.raises(FileNotFoundError, match="natural"):
        build_datasets(training_config(), base, root, seed=42)


def test_stratified_sample_keeps_proportions():
    df = _frame("x", 100)
    sample = stratified_sample(df, 60, seed=1, label_col="Label")
    assert sample["Label"].value_counts().to_dict() == {l: 20 for l in LABELS}
    assert len(stratified_sample(df, None, 1, "Label")) == 300
    assert len(stratified_sample(df, 10_000, 1, "Label")) == 300


class FakeTokenizer:
    """Whitespace tokenizer with the call signature of a Hugging Face tokenizer."""

    def __call__(self, texts, truncation=False, max_length=None):
        ids = [list(range(len(t.split()))) for t in texts]
        if truncation and max_length:
            ids = [i[:max_length] for i in ids]
        return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}


def test_encode_dataframe_and_dataset():
    df = pd.DataFrame({"Text": ["a b c", "d e"], "Label": ["positive", "negative"]})
    ds = encode_dataframe(FakeTokenizer(), df, LABEL_TO_ID, max_length=2, text_col="Text", label_col="Label")
    assert len(ds) == 2
    assert ds[0] == {"input_ids": [0, 1], "attention_mask": [1, 1], "labels": 2}  # truncated to 2 tokens
    assert ds[1]["labels"] == 0
    with pytest.raises(ValueError):
        EncodedDataset({"input_ids": [[1]]}, [0, 1])


def test_token_length_stats():
    texts = [" ".join(["w"] * n) for n in range(1, 101)]
    stats = token_length_stats(FakeTokenizer(), texts, max_length=90, seed=1)
    assert stats["max"] == 100 and stats["p50"] == 51 and stats["p95"] == 96
    assert stats["truncated_share"] == pytest.approx(0.10)


V4_FIELDS = {"output_dir", "evaluation_strategy", "warmup_ratio", "save_strategy", "load_best_model_at_end",
             "metric_for_best_model", "greater_is_better", "save_total_limit", "logging_steps",
             "dataloader_num_workers", "fp16", "seed", "data_seed", "report_to", "learning_rate",
             "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps",
             "num_train_epochs", "weight_decay"}
V5_FIELDS = (V4_FIELDS - {"evaluation_strategy", "warmup_ratio"}) | {"eval_strategy", "warmup_steps"}


def test_training_kwargs_adapt_to_transformers_versions():
    cfg = training_config()
    v4 = build_training_kwargs(cfg, 42, "out", V4_FIELDS, use_fp16=False)
    assert v4["evaluation_strategy"] == "epoch" and v4["warmup_ratio"] == 0.1 and "warmup_steps" not in v4
    v5 = build_training_kwargs(cfg, 42, "out", V5_FIELDS, use_fp16=True)
    assert v5["eval_strategy"] == "epoch" and v5["warmup_steps"] == 0.1 and "warmup_ratio" not in v5
    assert v5["fp16"] is True and v5["seed"] == 42 and v5["data_seed"] == 42
    assert v5["metric_for_best_model"] == "macro_f1" and v5["load_best_model_at_end"] is True

    smoke = build_training_kwargs(cfg, 42, "out", V5_FIELDS, use_fp16=False, smoke=True)
    assert smoke["num_train_epochs"] == 1.0 and smoke["per_device_train_batch_size"] == 4

    with pytest.raises(ConfigError, match="does not support"):
        build_training_kwargs(cfg, 42, "out", V5_FIELDS - {"data_seed"}, use_fp16=False)


def test_training_config_validation(tmp_path):
    validate_training_config(training_config())
    bad_cases = [
        training_config(train_on=[]),
        training_config(validate_on=["nope"]),
        training_config(tokenizer={"max_length": 2}),
        training_config(repeat={"natural": 0}),
        training_config(repeat={"unlisted": 2}),
        training_config(max_rows={"synthetic": -5}),
    ]
    for cfg in bad_cases:
        with pytest.raises(ConfigError):
            validate_training_config(cfg)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"model_name": "x"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing required"):
        load_training_config(path)


def test_shipped_xlmr_config_is_valid():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_training_config(root / "configs" / "xlm_roberta_base.yaml")
    base = load_base_config(root / "configs" / "base.yaml")
    assert cfg["tokenizer"]["max_length"] >= 8 and len(base["labels"]) == 3


def test_jsonl_log_callback_records_events(tmp_path):
    path = tmp_path / "logs" / "training_log.jsonl"
    callback = JsonlLogCallback(path)
    state = types.SimpleNamespace(global_step=10, epoch=1.0, best_model_checkpoint="ckpt-10", best_metric=0.9)
    callback.on_log(None, state, None, logs={"loss": 0.5})
    callback.on_evaluate(None, state, None, metrics={"eval_macro_f1": 0.9})
    callback.on_train_end(None, state, None)
    callback.on_log(None, state, None, logs=None)  # ignored
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert [r["event"] for r in records] == ["log", "evaluate", "train_end"]
    assert records[1]["eval_macro_f1"] == 0.9 and records[2]["best_checkpoint"] == "ckpt-10"