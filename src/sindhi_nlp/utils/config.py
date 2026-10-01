"""Read and validate YAML configuration files."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

DUPLICATE_POLICIES = ("drop", "report")
SPLIT_NAMES = ("train", "validation", "test")

REQUIRED_BASE_KEYS = (
    "seed",
    "labels",
    "label_mapping",
    "columns.text",
    "columns.label",
    "paths.raw_dir",
    "paths.interim_dir",
    "paths.processed_dir",
    "paths.reports_dir",
    "data.split.train",
    "data.split.validation",
    "data.split.test",
    "data.duplicate_policy",
)


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        try:
            data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def merge_configs(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = ...) -> Any:
    node: Any = config
    for part in dotted_key.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif default is ...:
            raise ConfigError(f"Missing config key: {dotted_key}")
        else:
            return default
    return node


def require_keys(config: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if get_nested(config, key, default=None) is None]
    if missing:
        raise ConfigError(f"Missing required config key(s): {', '.join(missing)}")


def load_config(path: str | Path, required: Iterable[str] = ()) -> dict[str, Any]:
    config = load_yaml(path)
    require_keys(config, required)
    return config


def validate_base_config(config: Mapping[str, Any]) -> None:
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"seed must be a non-negative integer, got {seed!r}")

    labels = config["labels"]
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(l, str) and l.strip() for l in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ConfigError("labels must be a list of unique, non-empty strings")

    mapping = config["label_mapping"]
    if not isinstance(mapping, Mapping) or set(mapping) != set(labels):
        raise ConfigError("label_mapping keys must match labels exactly")
    values = list(mapping.values())
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values) or len(set(values)) != len(values):
        raise ConfigError("label_mapping values must be unique integers")

    split = config["data"]["split"]
    ratios = [split[name] for name in SPLIT_NAMES]
    if not all(isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0 for r in ratios):
        raise ConfigError("data.split ratios must be positive numbers")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ConfigError(f"data.split ratios must sum to 1, got {sum(ratios)}")

    policy = config["data"]["duplicate_policy"]
    if policy not in DUPLICATE_POLICIES:
        raise ConfigError(f"data.duplicate_policy must be one of {DUPLICATE_POLICIES}, got {policy!r}")


def load_base_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path, required=REQUIRED_BASE_KEYS)
    validate_base_config(config)
    return config