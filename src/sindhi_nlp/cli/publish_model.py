"""Publish a trained model folder to the Hugging Face Hub.

The default is a dry run that only shows what would be uploaded:

    python -m sindhi_nlp.cli.publish_model --model-dir artifacts/models/xlmr_mixed

Add ``--upload`` to create the repository (private) and upload, and ``--public`` to make it public.
Authenticate first with ``hf auth login`` or the HF_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_REPO_ID = "alinawazmahar/sindhi-sentiment-xlmr"

# Only these files are uploaded. Everything else in the folder is left out on purpose.
REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "label_mapping.json",
    "preprocessing.json",
)
OPTIONAL_FILES = ("training_config.json", "metrics.json")
# training_args.bin is a pickle file (can run code when loaded); the log and checkpoints are clutter.
LEFT_OUT = ("training_args.bin", "training_log.jsonl", "checkpoints")

LOCAL_PATH = re.compile(r"[A-Za-z]:\\+(?:Users|Documents)|/home/[\w.-]+|/Users/[\w.-]+|/kaggle/|/tmp/")
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")

FRONT_MATTER = """---
language: sd
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: FacebookAI/xlm-roberta-base
tags:
  - sindhi
  - sentiment-analysis
  - text-classification
  - xlm-roberta
---

"""


class PublishError(RuntimeError):
    """The model folder is not ready to publish, or the upload cannot proceed."""


def explain_login_check_failure(exc: Exception) -> str:
    """Tell a login problem apart from a network problem, so the advice matches the cause."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403) or type(exc).__name__ == "LocalTokenNotFoundError":
        return "Not logged in to Hugging Face, or the token was rejected. Run `hf auth login` or set HF_TOKEN."
    return (
        f"Could not reach Hugging Face ({type(exc).__name__}: {exc}). "
        "This is usually a temporary network problem, so try again. Nothing was uploaded."
    )


@dataclass
class UploadPlan:
    """The files that will be uploaded to a Hugging Face model repository, and the repository they go to."""

    repo_id: str
    files: list[Path] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    readme: str = ""

    @property
    def total_bytes(self) -> int:
        """Total size of all files in the plan."""
        return sum(path.stat().st_size for path in self.files)


def build_readme(model_card: str) -> str:
    """The repository's README.md: Hub metadata followed by the project's model card."""
    if model_card.lstrip().startswith("---"):
        return model_card  # already has front matter
    return FRONT_MATTER + model_card.lstrip()


def find_local_paths(files: list[Path], readme: str) -> dict[str, list[str]]:
    """Personal or machine-specific paths hiding in the text files that would be published."""
    hits: dict[str, list[str]] = {}
    texts = {path.name: path.read_text("utf-8", errors="replace") for path in files if path.suffix == ".json"}
    texts["README.md"] = readme
    for name, text in texts.items():
        found = sorted(set(LOCAL_PATH.findall(text)))
        if found:
            hits[name] = found
    return hits


def plan_upload(model_dir: Path, model_card_path: Path, repo_id: str) -> UploadPlan:
    """Decide which files to publish: weights, tokenizer, configs and metrics, never pickle files or paths from your machine."""
    if not REPO_ID.match(repo_id):
        raise PublishError(f"The repository name must look like owner/name, got {repo_id!r}")
    if not model_dir.is_dir():
        raise PublishError(f"Model folder not found: {model_dir}")
    missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        hint = ""
        if "preprocessing.json" in missing:
            hint = " (create preprocessing.json with: sindhi-nlp finalize-model --model-dir <folder>)"
        raise PublishError(f"The model folder is missing: {', '.join(missing)}{hint}")
    if not model_card_path.is_file():
        raise PublishError(f"Model card not found: {model_card_path}")

    files = [model_dir / name for name in (*REQUIRED_FILES, *OPTIONAL_FILES) if (model_dir / name).is_file()]
    included = {path.name for path in files}
    left_out = sorted(entry.name for entry in model_dir.iterdir() if entry.name not in included)
    return UploadPlan(repo_id, files, left_out, build_readme(model_card_path.read_text("utf-8")))


def describe(plan: UploadPlan) -> str:
    """Describe an upload plan in plain text, for the dry run."""
    lines = [f"Repository: {plan.repo_id}", "Files to upload:"]
    for path in plan.files:
        lines.append(f"  {path.name:<26} {path.stat().st_size / 1e6:>9.2f} MB")
    lines.append(f"  {'README.md (model card)':<26} {len(plan.readme.encode('utf-8')) / 1e6:>9.2f} MB")
    lines.append(f"Total: {plan.total_bytes / 1e6:.1f} MB")
    if plan.left_out:
        lines.append("Left out on purpose: " + ", ".join(plan.left_out))
    return "\n".join(lines)


def publish(
    model_dir: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    model_card: str | Path = "docs/model_card.md",
    upload: bool = False,
    public: bool = False,
    allow_local_paths: bool = False,
    yes: bool = False,
    message: str = "Upload sindhi-nlp sentiment model",
    api: Any = None,
    confirm: Callable[[str], str] = input,
) -> dict[str, Any]:
    """Validate and (optionally) upload. Returns a summary; raises ``PublishError`` if something is wrong."""
    plan = plan_upload(Path(model_dir), Path(model_card), repo_id)

    hits = find_local_paths(plan.files, plan.readme)
    if hits and not allow_local_paths:
        details = "; ".join(f"{name}: {', '.join(found)}" for name, found in hits.items())
        raise PublishError(
            f"These files contain machine-specific paths and would publish them: {details}. "
            "Remove them, or pass --allow-local-paths if that is intended."
        )

    summary: dict[str, Any] = {"uploaded": False, "plan": plan, "url": f"https://huggingface.co/{repo_id}"}
    if not upload:
        return summary

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise PublishError("Uploading needs: pip install huggingface_hub") from exc
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    try:
        account = api.whoami()
    except Exception as exc:
        raise PublishError(explain_login_check_failure(exc)) from exc
    owners = {account.get("name")} | {org.get("name") for org in account.get("orgs", [])}
    owner = repo_id.split("/")[0]
    if owner not in owners:
        raise PublishError(f"You are logged in as {account.get('name')!r}, which cannot publish to {owner!r}")

    if public and not yes:
        answer = confirm(f"This makes {repo_id} PUBLIC. Type the repository name to confirm: ")
        if answer.strip() != repo_id:
            raise PublishError("Not confirmed; nothing was uploaded")

    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        for path in plan.files:
            shutil.copy2(path, staging / path.name)
        (staging / "README.md").write_text(plan.readme, encoding="utf-8", newline="\n")
        api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
        if public:
            api.update_repo_settings(
                repo_id=repo_id, private=False, repo_type="model"
            )  # for a repo that already existed
        api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="model", commit_message=message)
    summary["uploaded"] = True
    summary["public"] = public
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the publishing tool. It is a dry run unless ``--upload`` is given."""
    parser = argparse.ArgumentParser(
        description="Publish a trained model to the Hugging Face Hub (dry run by default)."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--model-card", default="docs/model_card.md")
    parser.add_argument("--upload", action="store_true", help="Really upload (default: dry run)")
    parser.add_argument("--public", action="store_true", help="Make the repository public (default: private)")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt for --public")
    parser.add_argument("--allow-local-paths", action="store_true")
    parser.add_argument("--message", default="Upload sindhi-nlp sentiment model")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    try:
        result = publish(
            root / args.model_dir,
            args.repo_id,
            root / args.model_card,
            args.upload,
            args.public,
            args.allow_local_paths,
            args.yes,
            args.message,
        )
    except PublishError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(describe(result["plan"]))
    if not result["uploaded"]:
        print("\nDRY RUN: nothing was uploaded. Add --upload to create a private repository and upload.")
        return 0
    state = "public" if result["public"] else "private (make it public with --public, or in the repository settings)"
    print(f"\nUploaded. The repository is {state}:\n  {result['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
