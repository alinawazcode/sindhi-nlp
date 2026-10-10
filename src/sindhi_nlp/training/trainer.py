"""XLM-RoBERTa fine-tuning: data assembly, tokenization, Trainer setup, saving.

Everything that needs torch/transformers is imported inside the functions that
use it, so the data-handling parts can be tested without those packages.
Only ``train.csv`` and ``validation.csv`` are ever read here; ``test.csv`` is
reserved for the final one-time evaluation.
"""
from __future__ import annotations

import dataclasses
import gc
import inspect
import platform
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.load_data import file_sha256, load_processed_split
from ..data.split_data import write_csv, write_json
from ..utils.config import ConfigError, load_base_config, load_config
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .callbacks import build_callbacks
from .metrics import compute_metrics, make_hf_compute_metrics

logger = get_logger(__name__)

ALLOWED_SPLITS = ("train", "validation")
REQUIRED_KEYS = (
    "model_name", "run_name", "datasets", "train_on", "validate_on", "tokenizer.max_length",
    "training.learning_rate", "training.num_train_epochs",
    "training.per_device_train_batch_size", "training.per_device_eval_batch_size",
)


# ----------------------------------------------------------------------------- config

def validate_training_config(cfg: Mapping[str, Any]) -> None:
    datasets = cfg["datasets"]
    if not isinstance(datasets, Mapping) or not datasets:
        raise ConfigError("datasets must map dataset names to folders")
    for key in ("train_on", "validate_on"):
        names = cfg[key]
        if not isinstance(names, list) or not names:
            raise ConfigError(f"{key} must be a non-empty list of dataset names")
    for key in ("train_on", "validate_on", "report_on"):
        unknown = [n for n in cfg.get(key) or [] if n not in datasets]
        if unknown:
            raise ConfigError(f"{key} names unknown dataset(s) {unknown}; known: {sorted(datasets)}")
    max_length = cfg["tokenizer"]["max_length"]
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 8:
        raise ConfigError("tokenizer.max_length must be an integer >= 8")
    for name, factor in (cfg.get("repeat") or {}).items():
        if name not in cfg["train_on"] or isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
            raise ConfigError(f"repeat[{name!r}] must be an integer >= 1 for a dataset listed in train_on")
    for name, rows in (cfg.get("max_rows") or {}).items():
        if name not in cfg["train_on"] or isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ConfigError(f"max_rows[{name!r}] must be a positive integer for a dataset listed in train_on")


def load_training_config(path: str | Path) -> dict[str, Any]:
    cfg = load_config(path, required=REQUIRED_KEYS)
    validate_training_config(cfg)
    return cfg


# ----------------------------------------------------------------------------- data

@dataclass
class DataBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    report: dict[str, pd.DataFrame] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    file_hashes: dict[str, str] = field(default_factory=dict)


def load_split(
    cfg: Mapping[str, Any], root: Path, source: str, split: str,
    labels: Sequence[str], text_col: str, label_col: str, hashes: dict[str, str] | None = None,
) -> pd.DataFrame:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Training only reads {ALLOWED_SPLITS}; the test split is reserved for the final evaluation")
    path = root / cfg["datasets"][source] / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Prepare the '{source}' dataset first.")
    if hashes is not None:
        hashes[f"{source}/{split}.csv"] = file_sha256(path)
    return load_processed_split(path, text_col, label_col, list(labels))


def stratified_sample(df: pd.DataFrame, n: int | None, seed: int, label_col: str) -> pd.DataFrame:
    """Up to ``n`` rows with the label proportions preserved."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    sample, _ = train_test_split(df, train_size=n, stratify=df[label_col], random_state=seed)
    return sample.reset_index(drop=True)


def build_datasets(
    cfg: Mapping[str, Any], base_cfg: Mapping[str, Any], root: Path, seed: int, smoke: bool = False
) -> DataBundle:
    labels = list(base_cfg["labels"])
    text_col, label_col = base_cfg["columns"]["text"], base_cfg["columns"]["label"]
    hashes: dict[str, str] = {}
    load = lambda source, split: load_split(cfg, root, source, split, labels, text_col, label_col, hashes)  # noqa: E731
    smoke_cfg = cfg.get("smoke") or {}

    validation = pd.concat([load(s, "validation") for s in cfg["validate_on"]], ignore_index=True)
    validation = validation.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    report: dict[str, pd.DataFrame] = {}
    for source in cfg.get("report_on") or []:
        if source in cfg["validate_on"]:
            continue
        try:
            report[source] = load(source, "validation")
        except FileNotFoundError as exc:
            logger.warning("Skipping report-only dataset '%s': %s", source, exc)

    parts = []
    for source in cfg["train_on"]:
        df = load(source, "train")
        df = stratified_sample(df, (cfg.get("max_rows") or {}).get(source), seed, label_col)
        parts.append(df.assign(_source=source))
    train = pd.concat(parts, ignore_index=True)

    before = len(train)
    train = train.drop_duplicates(subset=[text_col], keep="first")
    removed_duplicates = before - len(train)

    # Texts used for checkpoint selection or reporting must never be trained on.
    held_out = set(validation[text_col]).union(*(set(df[text_col]) for df in report.values()))
    before = len(train)
    train = train[~train[text_col].isin(held_out)]
    removed_overlap = before - len(train)
    if removed_overlap:
        logger.warning("Removed %d training rows whose text also appears in a validation set", removed_overlap)

    repeats = cfg.get("repeat") or {}
    train = pd.concat(
        [part for source, group in train.groupby("_source", sort=False) for part in [group] * int(repeats.get(source, 1))],
        ignore_index=True,
    )
    train = train.drop(columns="_source").sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if smoke:
        train = stratified_sample(train, int(smoke_cfg.get("train_rows", 240)), seed, label_col)
        validation = stratified_sample(validation, int(smoke_cfg.get("validation_rows", 90)), seed, label_col)
        report = {k: stratified_sample(v, int(smoke_cfg.get("validation_rows", 90)), seed, label_col) for k, v in report.items()}

    info = {
        "smoke": smoke,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "report_rows": {k: int(len(v)) for k, v in report.items()},
        "train_label_counts": {l: int((train[label_col] == l).sum()) for l in labels},
        "removed_cross_dataset_duplicates": int(removed_duplicates),
        "removed_validation_overlap": int(removed_overlap),
    }
    return DataBundle(train, validation, report, info, hashes)


try:
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:  # data handling stays usable (and testable) without torch
    _DatasetBase = object  # type: ignore[assignment,misc]


class EncodedDataset(_DatasetBase):
    """Map-style dataset: ``__len__`` and ``__getitem__`` returning a dict per row."""

    def __init__(self, encodings: Mapping[str, Sequence[Any]], labels: Sequence[int]) -> None:
        self.encodings = {k: list(v) for k, v in encodings.items()}
        self.labels = [int(l) for l in labels]
        if any(len(v) != len(self.labels) for v in self.encodings.values()):
            raise ValueError("encodings and labels have different lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def encode_dataframe(
    tokenizer: Callable[..., Any], df: pd.DataFrame, label_to_id: Mapping[str, int],
    max_length: int, text_col: str, label_col: str,
) -> EncodedDataset:
    """Tokenize without padding (a collator pads each batch dynamically)."""
    encodings = tokenizer(df[text_col].tolist(), truncation=True, max_length=max_length)
    return EncodedDataset(dict(encodings.items()), df[label_col].map(label_to_id).tolist())


def token_length_stats(
    tokenizer: Callable[..., Any], texts: Sequence[str], max_length: int, seed: int, sample_size: int = 5000
) -> dict[str, Any]:
    """Token-length percentiles on a sample, to sanity-check ``max_length``."""
    texts = list(texts)
    sample = texts if len(texts) <= sample_size else random.Random(seed).sample(texts, sample_size)
    lengths = sorted(len(ids) for ids in tokenizer(sample, truncation=False)["input_ids"])
    n = len(lengths)
    pct = lambda p: int(lengths[min(n - 1, int(p / 100 * n))])  # noqa: E731
    return {
        "sampled": n, "p50": pct(50), "p95": pct(95), "p99": pct(99), "max": int(lengths[-1]),
        "truncated_share": round(sum(l > max_length for l in lengths) / n, 6),
    }


# ----------------------------------------------------------------------------- Trainer setup

def build_training_kwargs(
    cfg: Mapping[str, Any], seed: int, output_dir: str, field_names: set[str], use_fp16: bool, smoke: bool = False,
) -> dict[str, Any]:
    """TrainingArguments keyword arguments that work across transformers versions."""
    t = dict(cfg["training"])
    smoke_cfg = cfg.get("smoke") or {}
    epochs = smoke_cfg.get("epochs", 1) if smoke else t["num_train_epochs"]
    batch = smoke_cfg.get("batch_size", 8) if smoke else t["per_device_train_batch_size"]
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": float(t["learning_rate"]),
        "per_device_train_batch_size": int(batch),
        "per_device_eval_batch_size": int(t["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(t.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(epochs),
        "weight_decay": float(t.get("weight_decay", 0.0)),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": int(t.get("save_total_limit", 2)),
        "logging_steps": 5 if smoke else int(t.get("logging_steps", 50)),
        "dataloader_num_workers": int(t.get("dataloader_num_workers", 0)),
        "fp16": bool(use_fp16),
        "seed": int(seed),
        "data_seed": int(seed),
        "report_to": "none",
    }
    # Renamed in transformers 4.41: evaluation_strategy -> eval_strategy.
    kwargs["eval_strategy" if "eval_strategy" in field_names else "evaluation_strategy"] = "epoch"
    # transformers 5 expresses a warmup ratio through a float warmup_steps below 1.
    warmup = float(t.get("warmup_ratio", 0.0))
    if "warmup_ratio" in field_names:
        kwargs["warmup_ratio"] = warmup
    else:
        kwargs["warmup_steps"] = warmup
    unknown = sorted(set(kwargs) - field_names)
    if unknown:
        raise ConfigError(f"This transformers version does not support: {unknown}")
    return kwargs


def _resolve_fp16(setting: Any) -> bool:
    if setting in (True, False):
        return bool(setting)
    import torch

    return bool(torch.cuda.is_available())


def _evaluate(trainer: Any, dataset: EncodedDataset, df: pd.DataFrame, labels: list[str],
              id2label: Mapping[int, str], label_col: str) -> tuple[dict[str, Any], np.ndarray]:
    output = trainer.predict(dataset)
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    predicted = np.array([id2label[i] for i in np.argmax(logits, axis=-1)], dtype=object)
    return compute_metrics(df[label_col], predicted, labels), predicted


def _remove_checkpoints(path: Path) -> None:
    """Delete intermediate checkpoints. On Windows, files can stay locked for a moment after loading."""
    for _ in range(3):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return
        gc.collect()
        time.sleep(0.5)
    logger.warning("Could not delete %s (files may be in use). It is safe to delete it manually.", path)


def format_run_summary(report: Mapping[str, Any]) -> str:
    lines = [
        f"# XLM-RoBERTa run: {report['run_name']}", "",
        f"- Base model: `{report['model_name']}`, seed {report['seed']}, smoke run: {report['data']['smoke']}",
        f"- Training rows: {report['data']['train_rows']:,} (label counts {report['data']['train_label_counts']})",
        f"- Best checkpoint metric (validation macro-F1): {report['best_metric']}",
        f"- Training time: {report['fit_seconds']:.0f}s", "",
        "| Dataset | Role | Rows | Macro-F1 | Accuracy |", "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, entry in report["evaluations"].items():
        m = entry["metrics"]
        lines.append(f"| {name} | {entry['role']} | {m['num_examples']:,} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |")
    lines += ["", "The test split was not used.", ""]
    return "\n".join(lines)


def train_model(
    config_path: str | Path = "configs/xlm_roberta_base.yaml",
    base_config_path: str | Path = "configs/base.yaml",
    project_root: str | Path = ".",
    smoke: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    base = load_base_config(base_config_path)
    cfg = load_training_config(config_path)

    seed = int(base["seed"])
    labels: list[str] = list(base["labels"])
    text_col, label_col = base["columns"]["text"], base["columns"]["label"]
    label_to_id = {label: int(base["label_mapping"][label]) for label in labels}
    id2label = {i: label for label, i in label_to_id.items()}

    paths = base["paths"]
    models_dir = root / paths.get("models_dir", "artifacts/models")
    metrics_dir = root / paths.get("metrics_dir", "artifacts/metrics")
    reports_dir = root / paths.get("model_reports_dir", "artifacts/reports")
    run_name = cfg["run_name"] + ("_smoke" if smoke else "")
    model_dir = models_dir / run_name
    if model_dir.exists() and any(model_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{model_dir} already exists. Choose a new run_name or pass overwrite.")
        shutil.rmtree(model_dir)

    data = build_datasets(cfg, base, root, seed, smoke)
    logger.info("Training rows: %d, validation rows: %d", len(data.train), len(data.validation))

    try:
        import torch
        import transformers
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Training needs torch and transformers: pip install torch transformers accelerate") from exc

    set_seed(seed)
    max_length = int(cfg["tokenizer"]["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    lengths = token_length_stats(tokenizer, data.train[text_col].tolist(), max_length, seed)
    logger.info("Token lengths: %s", lengths)
    if lengths["truncated_share"] > 0.01:
        logger.warning("%.1f%% of texts exceed max_length=%d and will be truncated",
                       100 * lengths["truncated_share"], max_length)

    encode = lambda df: encode_dataframe(tokenizer, df, label_to_id, max_length, text_col, label_col)  # noqa: E731
    train_ds, val_ds = encode(data.train), encode(data.validation)
    report_ds = {name: encode(df) for name, df in data.report.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(labels), id2label=id2label, label2id=label_to_id
    )
    field_names = {f.name for f in dataclasses.fields(TrainingArguments)}
    checkpoints = model_dir / "checkpoints"
    args = TrainingArguments(**build_training_kwargs(
        cfg, seed, str(checkpoints), field_names, _resolve_fp16(cfg["training"].get("fp16", "auto")), smoke
    ))
    log_path = model_dir / "training_log.jsonl"
    patience = 0 if smoke else int(cfg["training"].get("early_stopping_patience", 0))
    trainer_kwargs: dict[str, Any] = dict(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_hf_compute_metrics([id2label[i] for i in sorted(id2label)]),
        callbacks=build_callbacks(patience, log_path),
    )
    # `tokenizer=` was replaced by `processing_class=` in recent transformers releases.
    tokenizer_arg = "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    trainer_kwargs[tokenizer_arg] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    start = time.perf_counter()
    trainer.train()
    fit_seconds = time.perf_counter() - start

    evaluations: dict[str, Any] = {}
    val_metrics, val_pred = _evaluate(trainer, val_ds, data.validation, labels, id2label, label_col)
    evaluations["+".join(cfg["validate_on"])] = {"role": "validation (checkpoint selection)", "metrics": val_metrics}
    for name, ds in report_ds.items():
        metrics, _ = _evaluate(trainer, ds, data.report[name], labels, id2label, label_col)
        evaluations[name] = {"role": "report only", "metrics": metrics}

    # One folder holds everything needed to reload and audit the model.
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    write_json(label_to_id, model_dir / "label_mapping.json")
    state = trainer.state
    report: dict[str, Any] = {
        "run_name": run_name, "model_name": cfg["model_name"], "seed": seed,
        "best_checkpoint": state.best_model_checkpoint and Path(state.best_model_checkpoint).name,
        "best_metric": state.best_metric, "epochs_run": state.epoch, "fit_seconds": round(fit_seconds, 1),
        "data": data.info, "token_lengths": lengths, "evaluations": evaluations,
        "log_history": state.log_history,
    }
    write_json(
        {"seed": seed, "config": dict(cfg), "labels": labels, "data": data.info, "data_files_sha256": data.file_hashes,
         "token_lengths": lengths,
         "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__}},
        model_dir / "training_config.json",
    )
    write_json({"seed": seed, "best_metric": state.best_metric, "evaluations": evaluations}, model_dir / "metrics.json")

    wrong = data.validation[label_col].to_numpy() != val_pred
    errors = pd.DataFrame({text_col: data.validation[text_col].to_numpy()[wrong],
                           "true_label": data.validation[label_col].to_numpy()[wrong],
                           "predicted_label": val_pred[wrong]})
    write_json(report, metrics_dir / f"{run_name}_validation.json")
    write_csv(errors, reports_dir / f"{run_name}_misclassified_validation.csv")
    (reports_dir / f"{run_name}_summary.md").write_text(format_run_summary(report), encoding="utf-8", newline="\n")

    _remove_checkpoints(checkpoints)
    logger.info("Saved model to %s", model_dir)
    return report