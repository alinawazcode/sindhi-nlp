"""End-to-end Trainer run with a tiny random model and a local tokenizer.

Skipped automatically unless torch, transformers, tokenizers and accelerate are
installed. It needs no internet access: the tokenizer and model are built here.
"""
import json

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("accelerate")

from conftest import LABELS, base_config  # noqa: E402
from sindhi_nlp.training.trainer import train_model  # noqa: E402

VOCAB = {
    "negative": ["bad", "sad", "poor", "awful"],
    "neutral": ["table", "city", "book", "door"],
    "positive": ["good", "happy", "great", "nice"],
}


def _split(n_per_label: int, offset: int) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        for i in range(n_per_label):
            words = [VOCAB[label][(i + offset) % 4], VOCAB[label][(i + offset + 1) % 4], f"w{(i * 7 + offset) % 50}"]
            rows.append((" ".join(words), label))
    return pd.DataFrame(rows, columns=["Text", "Label"]).drop_duplicates("Text")


@pytest.fixture
def tiny_project(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification

    train, validation = _split(60, 0), _split(12, 3)
    validation = validation[~validation["Text"].isin(set(train["Text"]))]
    for folder in ("data/processed", "data/external/processed"):
        (tmp_path / folder).mkdir(parents=True)
        train.to_csv(tmp_path / folder / "train.csv", index=False, encoding="utf-8")
        validation.to_csv(tmp_path / folder / "validation.csv", index=False, encoding="utf-8")

    corpus = list(train["Text"]) + list(validation["Text"])
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=["<pad>", "<unk>", "<s>", "</s>"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>",
                                   bos_token="<s>", eos_token="</s>")
    model_path = tmp_path / "tiny_base"
    config = XLMRobertaConfig(vocab_size=len(fast), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                              intermediate_size=64, max_position_embeddings=64, pad_token_id=fast.pad_token_id,
                              bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id)
    XLMRobertaForSequenceClassification(config).save_pretrained(model_path)
    fast.save_pretrained(model_path)

    base = base_config()
    base["paths"].update({"models_dir": "artifacts/models", "metrics_dir": "artifacts/metrics",
                          "model_reports_dir": "artifacts/reports"})
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    cfg = {
        "model_name": str(model_path), "run_name": "tiny",
        "datasets": {"synthetic": "data/processed", "natural": "data/external/processed"},
        "train_on": ["synthetic"], "validate_on": ["natural"], "report_on": [],
        "tokenizer": {"max_length": 16},
        "training": {"learning_rate": 1.0e-3, "per_device_train_batch_size": 16, "per_device_eval_batch_size": 32,
                     "num_train_epochs": 2, "warmup_ratio": 0.1, "early_stopping_patience": 0, "fp16": False},
        "smoke": {"train_rows": 90, "validation_rows": 30, "epochs": 1, "batch_size": 16},
    }
    (tmp_path / "xlmr.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp_path


def test_full_pipeline_with_tiny_model(tiny_project):
    root = tiny_project
    report = train_model(root / "xlmr.yaml", root / "base.yaml", root)

    model_dir = root / "artifacts" / "models" / "tiny"
    for name in ("label_mapping.json", "training_config.json", "metrics.json", "training_log.jsonl", "config.json"):
        assert (model_dir / name).is_file(), name
    assert any(p.name.startswith("model") for p in model_dir.iterdir()), "model weights missing"
    assert not (model_dir / "checkpoints").exists()
    assert json.loads((model_dir / "training_config.json").read_text("utf-8"))["seed"] == 42
    assert 0.0 <= report["evaluations"]["natural"]["metrics"]["macro_f1"] <= 1.0
    assert (root / "artifacts" / "metrics" / "tiny_validation.json").is_file()
    assert (root / "artifacts" / "reports" / "tiny_summary.md").is_file()

    with pytest.raises(FileExistsError):
        train_model(root / "xlmr.yaml", root / "base.yaml", root)


def test_smoke_run_and_external_evaluation(tiny_project):
    from sindhi_nlp.evaluation.evaluate_external import evaluate_model

    root = tiny_project
    train_model(root / "xlmr.yaml", root / "base.yaml", root, smoke=True)
    smoke_dir = root / "artifacts" / "models" / "tiny_smoke"
    assert smoke_dir.is_dir()
    report = evaluate_model(smoke_dir, "data/external/processed/validation.csv", "validation",
                            base_config_path=root / "base.yaml", project_root=root)
    assert report["metrics"]["num_examples"] > 0

    