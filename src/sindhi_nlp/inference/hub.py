"""Load a published model by its Hugging Face name, for example ``owner/model-name``.

A local folder always wins over a Hub name, and ``pathlib.Path`` arguments are
never treated as Hub names, so a typo in a local path cannot trigger a download.
"""

from __future__ import annotations

import re
from pathlib import Path

from .schemas import ModelLoadError, ModelNotFoundError

# Weights and small config files only. Pickle files such as *.bin are never downloaded.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt"]
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def looks_like_repo_id(name: str) -> bool:
    """True for ``owner/name``. Paths such as ``a/b/c``, ``./a/b``, ``C:\\a\\b`` or ``/a/b`` are not repo ids."""
    return bool(_REPO_ID.match(name))


def download_model(repo_id: str, revision: str | None = None, cache_dir: str | Path | None = None) -> Path:
    """Download (or reuse the cached copy of) a model repository and return its local folder."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelLoadError("Loading a model by name needs: pip install huggingface_hub") from exc
    try:
        local = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception as exc:  # unknown repo, no network, no access
        raise ModelNotFoundError(
            f"{repo_id!r} is not a local folder and could not be downloaded from Hugging Face: {exc}"
        ) from exc
    return Path(local)


def resolve_model_path(model: str | Path, revision: str | None = None, cache_dir: str | Path | None = None) -> Path:
    """Return a local model folder for a path or a Hugging Face name."""
    if isinstance(model, Path) or Path(model).is_dir():
        return Path(model)
    if looks_like_repo_id(model):
        return download_model(model, revision, cache_dir)
    return Path(model)  # the caller reports "folder not found" for a path that does not exist
