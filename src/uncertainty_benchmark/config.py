"""Configuration loading, defaults, and validation."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = {
    "experiment_name": "uncertainty_experiment",
    "task_type": "binary_classification",
    "folds": {
        "n_folds": 1,
        "fold_ids": None,
    },
    "labels": {
        "classes": ["simple", "complex"],
    },
    "mc_dropout": {
        "committee_size": 20,
        "dropout_p": 0.10,
    },
    "embeddings": {
        "batch_size": 32,
    },
    "metrics": {
        "ece_bins": 15,
        "ti_fixed_coverage": 0.95,
    },
    "methods": {
        "enabled": ["SR", "ENT"],
    },
    "outputs": {
        "outdir": "./results/uncertainty_experiment",
        "save_per_fold_scores": True,
        "save_wide_scores": True,
        "save_long_scores": True,
    },
}


VALID_METHODS = {
    "SR",
    "ENT",
    "SMP",
    "PV",
    "BALD",
    "ENT_MC",
    "MD",
    "HUQ-MD",
    "LOF",
    "ISOF",
}


REQUIRED_TOP_LEVEL_KEYS = {
    "model",
    "data",
}


REQUIRED_MODEL_KEYS = {
    "model_id_template",
}


REQUIRED_DATA_KEYS = {
    "train_file",
    "pred_csv_template",
    "lang_name",
}


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively update a dictionary."""
    out = deepcopy(base)

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value

    return out


def expand_path_string(value: str) -> str:
    """Expand environment variables and user home in path-like strings."""
    return os.path.expandvars(os.path.expanduser(value))


def normalise_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply defaults and normalise paths."""
    cfg = deep_update(DEFAULT_CONFIG, config)

    # If no explicit output directory was supplied, use experiment name.
    if not config.get("outputs", {}).get("outdir"):
        cfg["outputs"]["outdir"] = f"./results/{cfg['experiment_name']}"

    for section, keys in {
        "data": ["train_file", "pred_csv_template"],
        "outputs": ["outdir"],
    }.items():
        if section in cfg:
            for key in keys:
                if key in cfg[section] and isinstance(cfg[section][key], str):
                    cfg[section][key] = expand_path_string(cfg[section][key])

    return cfg


def resolve_fold_ids_from_config(config: dict) -> list[int]:
    """Resolve fold ids from config only.

    This is duplicated lightly from runner-level logic so config validation
    can check folds before execution.
    """
    folds = config.get("folds", {})

    fold_ids = folds.get("fold_ids")

    if fold_ids is not None:
        if not isinstance(fold_ids, list):
            raise TypeError("folds.fold_ids must be a list of integers or null.")
        return [int(x) for x in fold_ids]

    n_folds = int(folds.get("n_folds", 1))
    if n_folds < 1:
        raise ValueError("folds.n_folds must be >= 1.")

    return list(range(n_folds))


def validate_config(config: dict, check_files: bool = False) -> None:
    """Validate a benchmark config.

    Parameters
    ----------
    config:
        Normalised config dictionary.

    check_files:
        If True, check that the training file and all prediction files exist.
        Keep False for tests or when creating configs before data is copied.
    """
    missing_top = REQUIRED_TOP_LEVEL_KEYS.difference(config.keys())
    if missing_top:
        raise KeyError(f"Missing required top-level config sections: {sorted(missing_top)}")

    missing_model = REQUIRED_MODEL_KEYS.difference(config["model"].keys())
    if missing_model:
        raise KeyError(f"Missing required model config keys: {sorted(missing_model)}")

    missing_data = REQUIRED_DATA_KEYS.difference(config["data"].keys())
    if missing_data:
        raise KeyError(f"Missing required data config keys: {sorted(missing_data)}")

    fold_ids = resolve_fold_ids_from_config(config)

    if len(fold_ids) == 0:
        raise ValueError("No folds selected. Set folds.fold_ids or folds.n_folds.")

    enabled_methods = config.get("methods", {}).get("enabled", [])

    if not enabled_methods:
        raise ValueError("methods.enabled cannot be empty.")

    unknown_methods = sorted(set(enabled_methods).difference(VALID_METHODS))
    if unknown_methods:
        raise ValueError(
            f"Unknown methods in config: {unknown_methods}. "
            f"Valid methods: {sorted(VALID_METHODS)}"
        )

    committee_size = int(config.get("mc_dropout", {}).get("committee_size", 20))
    if committee_size < 1:
        raise ValueError("mc_dropout.committee_size must be >= 1.")

    dropout_p = float(config.get("mc_dropout", {}).get("dropout_p", 0.10))
    if not (0.0 <= dropout_p <= 1.0):
        raise ValueError("mc_dropout.dropout_p must be between 0 and 1.")

    ece_bins = int(config.get("metrics", {}).get("ece_bins", 15))
    if ece_bins < 1:
        raise ValueError("metrics.ece_bins must be >= 1.")

    ti_cov = float(config.get("metrics", {}).get("ti_fixed_coverage", 0.95))
    if not (0.0 < ti_cov <= 1.0):
        raise ValueError("metrics.ti_fixed_coverage must be in (0, 1].")

    if check_files:
        train_file = Path(config["data"]["train_file"])
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")

        pred_template = config["data"]["pred_csv_template"]
        for fold_id in fold_ids:
            pred_file = Path(pred_template.format(fold_id=fold_id))
            if not pred_file.exists():
                raise FileNotFoundError(
                    f"Prediction file not found for fold {fold_id}: {pred_file}"
                )


def load_config(path, check_files: bool = False) -> dict:
    """Load, normalise, and validate YAML config."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    cfg = normalise_config(raw)
    validate_config(cfg, check_files=check_files)

    return cfg


def save_config(config: dict, path) -> Path:
    """Save config dictionary to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    return path
