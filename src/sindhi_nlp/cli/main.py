"""Terminal commands for Sindhi sentiment prediction.

    python -m sindhi_nlp.cli.main predict "اڄ جو ڏينهن تمام سٺو آهي"
    python -m sindhi_nlp.cli.main batch --input texts.csv --output results.csv
    python -m sindhi_nlp.cli.main info
    python -m sindhi_nlp.cli.main finalize-model --model-dir artifacts/models/xlmr_mixed

Run from the project root with PYTHONPATH=src (or after installing the package).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.clean_data import CleaningOptions
from ..data.load_data import read_csv
from ..inference.predictor import SentimentPredictor
from ..inference.preprocessing import write_preprocessing_config
from ..inference.schemas import InvalidInputError, ModelLoadError, ModelNotFoundError
from ..utils.config import ConfigError, load_config
from ..utils.logging import setup_logging

DEFAULT_CONFIG = "configs/inference.yaml"
DEFAULT_BASE_CONFIG = "configs/base.yaml"
KNOWN_ERRORS = (ModelNotFoundError, ModelLoadError, InvalidInputError, ConfigError, FileNotFoundError, ValueError)


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", default=None, help="Model folder (default: model_dir from the config)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Inference config (default: configs/inference.yaml)")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sindhi-nlp", description="Sindhi sentiment classification")
    parser.add_argument("--log-level", default="WARNING")
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict", help="Classify one or more texts")
    predict.add_argument("texts", nargs="+", help="Text(s) to classify")
    predict.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    _add_model_options(predict)

    batch = commands.add_parser("batch", help="Classify a CSV (or a text file with one sentence per line)")
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", required=True, help="CSV to write")
    batch.add_argument("--text-column", default="Text", help="Column holding the text (CSV input)")
    _add_model_options(batch)

    info = commands.add_parser("info", help="Show model details")
    info.add_argument("--json", action="store_true")
    _add_model_options(info)

    finalize = commands.add_parser(
        "finalize-model", help="Save the text-cleaning options next to a model (writes preprocessing.json)"
    )
    finalize.add_argument("--model-dir", required=True)
    finalize.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    return parser


def load_predictor(args: argparse.Namespace) -> SentimentPredictor:
    """Build the predictor from --model-dir, or from the inference config."""
    root = Path(args.project_root).resolve()
    if args.model_dir:
        return SentimentPredictor.from_pretrained(root / args.model_dir, device=args.device or "auto")
    config_path = root / args.config
    if not config_path.is_file():
        raise ConfigError(f"No --model-dir given and {config_path} does not exist")
    cfg = load_config(config_path, required=("model_dir",))
    return SentimentPredictor.from_pretrained(
        root / cfg["model_dir"],
        device=args.device or cfg.get("device", "auto"),
        batch_size=int(cfg.get("batch_size") or 32),
        max_length=cfg.get("max_length"),
        confidence_threshold=cfg.get("confidence_threshold"),
        max_chars=cfg.get("max_chars"),
    )


def cmd_predict(args: argparse.Namespace) -> int:
    predictor = load_predictor(args)
    results = predictor.predict_batch(args.texts, errors="none")
    if args.json:
        payload: list[dict[str, Any] | None] = [r.to_dict() if r else None for r in results]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for text, result in zip(args.texts, results):
            if result is None:
                print(f"{'invalid':<9} {'-':>5}  {text!r}")
            else:
                flag = "" if result.is_confident else "  (low confidence)"
                print(f"{result.label:<9} {result.confidence:>5.3f}  {text}{flag}")
    return 1 if any(r is None for r in results) else 0


def read_batch_input(path: Path, text_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = read_csv(path)
        lowered = {c.lower(): c for c in df.columns}
        column = lowered.get(text_column.lower())
        if column is None:
            raise ValueError(f"Column {text_column!r} not found; the file has {list(df.columns)}")
        return df.rename(columns={column: text_column}) if column != text_column else df
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return pd.DataFrame({text_column: lines})


def cmd_batch(args: argparse.Namespace) -> int:
    predictor = load_predictor(args)
    df = read_batch_input(Path(args.project_root).resolve() / args.input, args.text_column)
    results = predictor.predict_batch(df[args.text_column].tolist(), errors="none")

    out = df.copy()
    out["predicted_label"] = [r.label if r else "" for r in results]
    out["confidence"] = [round(r.confidence, 6) if r else "" for r in results]
    for label in predictor.labels:
        out[f"prob_{label}"] = [round(r.probabilities[label], 6) if r else "" for r in results]
    out["error"] = ["" if r else "invalid input" for r in results]

    target = Path(args.project_root).resolve() / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, encoding="utf-8-sig", lineterminator="\n")  # BOM lets Excel show Sindhi correctly
    invalid = sum(r is None for r in results)
    print(f"Classified {len(results) - invalid:,} of {len(results):,} rows -> {target}")
    if invalid:
        print(f"{invalid:,} row(s) were empty or invalid and were left blank", file=sys.stderr)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    info = load_predictor(args).info
    if args.json:
        print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
    else:
        for key, value in info.to_dict().items():
            print(f"{key:<22} {value}")
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    cfg = load_config(args.base_config)
    options = CleaningOptions.from_config((cfg.get("data") or {}).get("cleaning"))
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise ModelNotFoundError(f"Model folder not found: {model_dir}")
    path = write_preprocessing_config(model_dir, options, source=f"{args.base_config} data.cleaning")
    print(f"Wrote {path}")
    return 0


COMMANDS = {"predict": cmd_predict, "batch": cmd_batch, "info": cmd_info, "finalize-model": cmd_finalize}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # Sindhi text must print on any Windows console
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return COMMANDS[args.command](args)
    except KNOWN_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())