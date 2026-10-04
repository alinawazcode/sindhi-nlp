"""Command-line entry point for XLM-RoBERTa fine-tuning.

    PYTHONPATH=src python -m sindhi_nlp.training.train --smoke      # quick pipeline check
    PYTHONPATH=src python -m sindhi_nlp.training.train              # full run (use a GPU)
"""
from __future__ import annotations

import argparse

from ..utils.logging import setup_logging
from .trainer import train_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune XLM-RoBERTa for Sindhi sentiment.")
    parser.add_argument("--config", default="configs/xlm_roberta_base.yaml")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to verify setup, saving, and evaluation")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing model folder")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    report = train_model(args.config, args.base_config, args.project_root, args.smoke, args.overwrite)

    print(f"Run: {report['run_name']}   best validation macro-F1: {report['best_metric']}")
    for name, entry in report["evaluations"].items():
        m = entry["metrics"]
        print(f"  {name:<20} {entry['role']:<34} macro-F1 {m['macro_f1']:.4f}  accuracy {m['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())