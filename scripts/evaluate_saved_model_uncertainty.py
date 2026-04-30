#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-training uncertainty evaluation from saved prediction CSVs.

This script is designed for the case where the models have already been
trained and the validation/test prediction files already exist.

What it does
------------
For each configured fold, it can:

1. Load a saved prediction CSV.
2. Standardise text, labels, predictions, and probability columns.
3. Compute deterministic uncertainty scores from saved probabilities:
   - SR      = 1 - max probability
   - ENT     = entropy of the predictive distribution
   - MARGIN  = 1 - absolute binary probability margin
4. Reuse already-saved uncertainty-score columns, e.g. SMP, PV, BALD, ENT_MC.
5. Optionally reconstruct the fold train pool by removing validation rows from
   the original train file, if ``data.train_file`` is provided.
6. Call the reusable metric suite from ``uncertainty_benchmark.metrics.suite``.
7. Save per-fold and all-fold metric files.

Important convention
--------------------
The metric suite expects every method column to follow this direction:

    larger score = more uncertain

If a saved score is a confidence score instead, set this in the YAML config:

methods:
  score_direction:
    MY_CONFIDENCE_SCORE: confidence

Then the script will convert it to uncertainty with ``1 - score`` before
metric computation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------
# Make local package imports robust when running from scripts/ or repo root
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
CANDIDATE_IMPORT_ROOTS = [
    Path.cwd(),
    Path.cwd() / "src",
    THIS_FILE.parent,
    THIS_FILE.parent.parent,
    THIS_FILE.parent.parent / "src",
]

for candidate in CANDIDATE_IMPORT_ROOTS:
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

try:
    from uncertainty_benchmark.metrics.suite import (
        compute_metrics_per_method_with_timing,
        metrics_to_long,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import uncertainty_benchmark. Run this script from the "
        "repository root, install the package with `pip install -e .`, or make "
        "sure the package folder / src folder is on PYTHONPATH."
    ) from exc


# ---------------------------------------------------------------------
# Config / path helpers
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty from saved fold prediction CSVs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(config, dict):
        raise TypeError(f"Config file must contain a YAML dictionary: {path}")

    return config


def resolve_path(path_like: str | Path | None) -> Path | None:
    """Resolve paths relative to the current working directory."""
    if path_like is None:
        return None

    path_str = str(path_like).strip()
    if not path_str:
        return None

    return Path(path_str).expanduser().resolve()


def format_template(template: str, fold_id: int) -> str:
    """Format a path template that contains ``{fold_id}``."""
    return template.format(fold_id=fold_id)


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


# ---------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------


def find_column(
    df: pd.DataFrame,
    candidates: list[str] | tuple[str, ...],
    purpose: str,
    required: bool = True,
) -> str | None:
    """Find the first matching column from candidate names.

    Matching is first exact and then case-insensitive.
    """
    candidates = [str(c) for c in as_list(candidates) if str(c).strip()]

    for col in candidates:
        if col in df.columns:
            return col

    lower_map = {str(col).lower(): col for col in df.columns}

    for col in candidates:
        key = col.lower()
        if key in lower_map:
            return lower_map[key]

    if required:
        raise KeyError(
            f"Could not find column for {purpose}. "
            f"Candidates: {candidates}. Available columns: {list(df.columns)}"
        )

    return None


def normalise_text(text: object, normalise_whitespace: bool = True) -> str:
    """Normalise text for matching validation rows to original rows."""
    if pd.isna(text):
        return ""

    text_str = str(text).replace("\ufeff", "")

    if normalise_whitespace:
        text_str = re.sub(r"\s+", " ", text_str)

    return text_str.strip()


def standardise_label(value: object, classes: list[str]) -> str:
    """Standardise string labels using the configured class names."""
    value_str = str(value).strip()

    for cls in classes:
        if value_str.lower() == str(cls).lower():
            return str(cls)

    return value_str


def make_safe_column_name(name: object) -> str:
    """Create a stable, simple method/column name."""
    name_str = str(name).strip()
    name_str = re.sub(r"\s+", "_", name_str)
    name_str = re.sub(r"[^0-9A-Za-z_\-]+", "_", name_str)
    return name_str.strip("_") or "score"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_table(path: Path) -> pd.DataFrame:
    """Load CSV or Excel file."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path}")


def load_original_train(
    config: dict[str, Any],
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """Load and standardise the original full train file, if provided.

    If ``data.train_file`` is not provided, the script runs in evaluation-only
    mode and skips train-pool reconstruction.
    """
    train_file = resolve_path(get_nested(config, "data", "train_file", default=None))
    require_train_file = bool(get_nested(config, "data", "require_train_file", default=False))

    if train_file is None:
        if require_train_file:
            raise ValueError("data.train_file is required but was not provided.")
        print("No data.train_file provided; running in evaluation-only mode.")
        return None, None, None

    if not train_file.exists():
        if require_train_file:
            raise FileNotFoundError(f"Required train file not found: {train_file}")
        print(f"Train file not found; skipping train-pool reconstruction: {train_file}")
        return None, None, None

    train_df = load_table(train_file)

    text_col = find_column(
        train_df,
        get_nested(config, "data", "train_text_column_candidates", default=[]),
        purpose="train text",
    )

    classes = get_nested(config, "labels", "classes", default=["simple", "complex"])

    rating_enabled = bool(
        get_nested(config, "data", "rating_to_label", "enabled", default=False)
    )

    train_df = train_df.copy()
    train_df["text"] = train_df[text_col].apply(normalise_text)

    if rating_enabled:
        rating_col = get_nested(
            config,
            "data",
            "rating_to_label",
            "rating_column",
            default="Rating",
        )
        threshold = get_nested(
            config,
            "data",
            "rating_to_label",
            "complex_if_greater_equal",
            default=4,
        )
        simple_label = get_nested(
            config,
            "data",
            "rating_to_label",
            "simple_label",
            default="simple",
        )
        complex_label = get_nested(
            config,
            "data",
            "rating_to_label",
            "complex_label",
            default="complex",
        )

        if rating_col not in train_df.columns:
            raise KeyError(
                f"Rating-to-label is enabled, but column '{rating_col}' "
                f"was not found in {train_file}."
            )

        ratings = pd.to_numeric(train_df[rating_col], errors="coerce")
        train_df["label"] = np.where(
            ratings >= threshold,
            complex_label,
            simple_label,
        )

        label_col = rating_col

    else:
        label_col = find_column(
            train_df,
            get_nested(config, "data", "train_label_column_candidates", default=[]),
            purpose="train label",
        )
        train_df["label"] = train_df[label_col].apply(
            lambda x: standardise_label(x, classes)
        )

    return train_df, text_col, label_col


def load_validation_predictions(
    config: dict[str, Any],
    fold_id: int,
) -> tuple[pd.DataFrame, Path]:
    """Load and standardise one validation prediction CSV."""
    pred_template = get_nested(config, "data", "pred_csv_template")
    if pred_template is None:
        raise ValueError("Missing required config key: data.pred_csv_template")

    pred_path = resolve_path(format_template(str(pred_template), fold_id))
    if pred_path is None:
        raise ValueError("Could not resolve data.pred_csv_template.")

    val_df = pd.read_csv(pred_path)

    eval_text_col = find_column(
        val_df,
        get_nested(config, "data", "eval_text_column_candidates", default=[]),
        purpose="validation text",
    )

    true_col = find_column(
        val_df,
        get_nested(config, "data", "eval_true_label_column_candidates", default=[]),
        purpose="validation true label",
    )

    pred_col = find_column(
        val_df,
        get_nested(config, "data", "eval_pred_label_column_candidates", default=[]),
        purpose="validation predicted label",
    )

    classes = get_nested(config, "labels", "classes", default=["simple", "complex"])
    lang_name = get_nested(config, "data", "lang_name", default=None)

    val_df = val_df.copy()

    if "Lang" in val_df.columns and lang_name is not None:
        before = len(val_df)
        val_df = val_df[val_df["Lang"].astype(str) == str(lang_name)].copy()
        print(f"Filtered Lang == {lang_name}: {before} -> {len(val_df)} rows")

    val_df["text"] = val_df[eval_text_col].apply(normalise_text)
    val_df["true_label"] = val_df[true_col].apply(
        lambda x: standardise_label(x, classes)
    )
    val_df["pred_label"] = val_df[pred_col].apply(
        lambda x: standardise_label(x, classes)
    )
    val_df["correct"] = (val_df["true_label"] == val_df["pred_label"]).astype(int)

    return val_df, pred_path


# ---------------------------------------------------------------------
# Probability handling
# ---------------------------------------------------------------------


def probability_column_candidates_for_class(
    config: dict[str, Any],
    cls: str,
) -> list[str]:
    """Build candidate raw probability column names for one class."""
    probability_columns = get_nested(config, "labels", "probability_columns", default=None)

    candidates: list[str] = []

    if isinstance(probability_columns, dict) and cls in probability_columns:
        candidates.extend([str(x) for x in as_list(probability_columns[cls])])

    prob_template = get_nested(
        config,
        "labels",
        "probability_column_template",
        default="Prob_{class}",
    )

    for template in as_list(prob_template):
        template_str = str(template)
        # Avoid ``template.format(class=...)`` because ``class`` is a Python keyword.
        candidates.append(template_str.replace("{class}", str(cls)))

    # Useful fallbacks across older result files.
    candidates.extend(
        [
            f"Prob_{cls}",
            f"prob_{cls}",
            f"P_{cls}",
            f"p_{cls}",
            f"probability_{cls}",
        ]
    )

    # Preserve order while removing duplicates.
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    return unique_candidates


def add_probability_columns(
    config: dict[str, Any],
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardise probability columns as ``prob_<class>``.

    Set ``labels.require_probability_columns: false`` to skip probability-based
    deterministic scores when probability columns are absent.
    """
    val_df = val_df.copy()

    classes = get_nested(config, "labels", "classes", default=["simple", "complex"])
    require_probs = bool(get_nested(config, "labels", "require_probability_columns", default=True))

    found_probs: list[str] = []
    missing_messages: list[str] = []

    for cls in classes:
        raw_col = find_column(
            val_df,
            probability_column_candidates_for_class(config, str(cls)),
            purpose=f"probability for class {cls}",
            required=False,
        )
        std_col = f"prob_{cls}"

        if raw_col is None:
            missing_messages.append(
                f"class={cls}; tried={probability_column_candidates_for_class(config, str(cls))}"
            )
            continue

        val_df[std_col] = pd.to_numeric(val_df[raw_col], errors="coerce")
        found_probs.append(std_col)

    if len(found_probs) != len(classes):
        message = (
            "Could not find all configured probability columns. "
            + " | ".join(missing_messages)
        )
        if require_probs:
            raise KeyError(message + f". Available columns: {list(val_df.columns)}")
        print("Warning:", message, "Probability-derived methods will be skipped.")
        return val_df

    prob_sum = val_df[found_probs].sum(axis=1)
    if len(prob_sum):
        max_deviation = float(np.nanmax(np.abs(prob_sum - 1.0)))
    else:
        max_deviation = float("nan")

    if np.isfinite(max_deviation) and max_deviation > 1e-3:
        print(
            "Warning: probability columns do not sum to 1. "
            f"Maximum deviation = {max_deviation:.6f}"
        )

    return val_df


def probability_matrix_from_val(
    config: dict[str, Any],
    val_df: pd.DataFrame,
) -> np.ndarray:
    """Return probability matrix in the order of ``labels.classes``."""
    classes = get_nested(config, "labels", "classes", default=["simple", "complex"])
    prob_cols = [f"prob_{cls}" for cls in classes]

    missing = [col for col in prob_cols if col not in val_df.columns]

    if missing:
        raise KeyError(f"Missing standardised probability columns: {missing}")

    return val_df[prob_cols].astype(float).to_numpy()


# ---------------------------------------------------------------------
# Label index handling
# ---------------------------------------------------------------------


def add_label_indices(
    config: dict[str, Any],
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add integer label columns required by the metric suite."""
    val_df = val_df.copy()

    classes = get_nested(config, "labels", "classes", default=["simple", "complex"])
    label_to_idx = {str(label): idx for idx, label in enumerate(classes)}

    unknown_true = sorted(set(val_df["true_label"]) - set(label_to_idx))
    unknown_pred = sorted(set(val_df["pred_label"]) - set(label_to_idx))

    if unknown_true:
        raise ValueError(f"Unknown true labels not in config labels.classes: {unknown_true}")

    if unknown_pred:
        raise ValueError(f"Unknown predicted labels not in config labels.classes: {unknown_pred}")

    val_df["y_true_idx"] = val_df["true_label"].map(label_to_idx).astype(int)
    val_df["y_pred_idx"] = val_df["pred_label"].map(label_to_idx).astype(int)

    return val_df


# ---------------------------------------------------------------------
# Uncertainty scores
# ---------------------------------------------------------------------


def add_deterministic_uncertainty_scores(
    config: dict[str, Any],
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Add SR, ENT, and MARGIN uncertainty scores from saved probabilities.

    SR:
        Softmax response uncertainty = 1 - max probability.

    ENT:
        Entropy of the deterministic predictive distribution.

    MARGIN:
        Binary probability-margin uncertainty = 1 - |p0 - p1|.
    """
    val_df = val_df.copy()
    enabled = [str(m) for m in get_nested(config, "methods", "enabled", default=[])]

    probability_methods = {"SR", "ENT", "MARGIN"}
    requested_probability_methods = [m for m in enabled if m in probability_methods]

    available_methods: list[str] = []

    if not requested_probability_methods:
        return val_df, available_methods

    try:
        probs = probability_matrix_from_val(config, val_df)
    except KeyError as exc:
        print(f"Skipping probability-derived methods {requested_probability_methods}: {exc}")
        return val_df, available_methods

    if "SR" in enabled:
        val_df["SR"] = 1.0 - np.max(probs, axis=1)
        available_methods.append("SR")

    if "ENT" in enabled:
        val_df["ENT"] = -np.sum(
            probs * np.log(np.clip(probs, 1e-8, 1.0)),
            axis=1,
        )
        available_methods.append("ENT")

    if "MARGIN" in enabled:
        if probs.shape[1] != 2:
            print("Skipping MARGIN because it is only implemented for binary classification.")
        else:
            val_df["MARGIN"] = 1.0 - np.abs(probs[:, 0] - probs[:, 1])
            available_methods.append("MARGIN")

    return val_df, available_methods


def saved_score_column_candidates(
    config: dict[str, Any],
    method: str,
) -> list[str]:
    """Return candidate columns for an already-saved score method."""
    method = str(method)

    candidates: list[str] = []

    explicit_map = get_nested(config, "methods", "saved_score_columns", default={})
    if isinstance(explicit_map, dict) and method in explicit_map:
        candidates.extend([str(x) for x in as_list(explicit_map[method])])

    templates = get_nested(
        config,
        "methods",
        "saved_score_column_templates",
        default=["{method}", "{method}_score", "score_{method}", "uncertainty_{method}"],
    )

    for template in as_list(templates):
        candidates.append(str(template).replace("{method}", method))

    lower = method.lower()
    upper = method.upper()
    candidates.extend(
        [
            method,
            lower,
            upper,
            f"{method}_uncertainty",
            f"{lower}_uncertainty",
            f"uncertainty_{method}",
            f"uncertainty_{lower}",
            f"{method}_score",
            f"{lower}_score",
        ]
    )

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    return unique_candidates


def add_existing_uncertainty_score_columns(
    config: dict[str, Any],
    val_df: pd.DataFrame,
    available_methods: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Reuse uncertainty-score columns that already exist in the CSV.

    This is what enables evaluation of saved MC-dropout scores such as SMP,
    PV, BALD, or ENT_MC without rerunning the model.
    """
    val_df = val_df.copy()
    enabled = [str(m) for m in get_nested(config, "methods", "enabled", default=[])]

    # These are generated from probabilities by add_deterministic_uncertainty_scores.
    generated_methods = {"SR", "ENT", "MARGIN"}

    for method in enabled:
        if method in available_methods:
            continue

        candidates = saved_score_column_candidates(config, method)
        raw_col = find_column(
            val_df,
            candidates,
            purpose=f"saved uncertainty score for {method}",
            required=False,
        )

        if raw_col is None:
            if method not in generated_methods:
                print(
                    f"Warning: method {method} is enabled but no saved score column was found. "
                    f"Tried: {candidates}"
                )
            continue

        method_col = make_safe_column_name(method)

        if raw_col != method_col:
            if method_col in val_df.columns:
                method_col = make_safe_column_name(f"{method}_from_{raw_col}")
            val_df[method_col] = pd.to_numeric(val_df[raw_col], errors="coerce")
        else:
            val_df[method_col] = pd.to_numeric(val_df[raw_col], errors="coerce")

        available_methods.append(method_col)
        print(f"Using saved score column for {method}: {raw_col} -> {method_col}")

    return val_df, available_methods


def score_columns_for_metrics(
    config: dict[str, Any],
    score_df: pd.DataFrame,
    methods: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare score columns so all methods follow larger = more uncertain.

    Confidence methods are inverted before metric computation.
    """
    score_df = score_df.copy()

    score_direction = get_nested(
        config,
        "methods",
        "score_direction",
        default={},
    )

    metric_methods: list[str] = []

    for method in methods:
        if method not in score_df.columns:
            continue

        # If method was renamed, allow lookup by original base name before suffixes.
        direction = "uncertainty"
        if isinstance(score_direction, dict):
            direction = score_direction.get(method, None)
            if direction is None:
                base = str(method).replace("_uncertainty", "")
                direction = score_direction.get(base, "uncertainty")

        if direction == "uncertainty":
            score_df[method] = pd.to_numeric(score_df[method], errors="coerce")
            metric_methods.append(method)

        elif direction == "confidence":
            metric_col = f"{method}_uncertainty"
            score_df[metric_col] = 1.0 - pd.to_numeric(score_df[method], errors="coerce")
            metric_methods.append(metric_col)

        else:
            raise ValueError(
                f"Unknown score direction for {method}: {direction}. "
                "Use 'uncertainty' or 'confidence'."
            )

    return score_df, metric_methods


# ---------------------------------------------------------------------
# Train-pool reconstruction
# ---------------------------------------------------------------------


def reconstruct_train_pool(
    config: dict[str, Any],
    train_df: pd.DataFrame | None,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Remove validation rows from the original train file, if available."""
    if train_df is None:
        return None, None

    removal_enabled = bool(
        get_nested(config, "data", "validation_removal", "enabled", default=True)
    )

    if not removal_enabled:
        return train_df.copy(), train_df.iloc[0:0].copy()

    normalise_ws = bool(
        get_nested(
            config,
            "data",
            "validation_removal",
            "normalise_whitespace",
            default=True,
        )
    )

    case_sensitive = bool(
        get_nested(
            config,
            "data",
            "validation_removal",
            "case_sensitive",
            default=True,
        )
    )

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_keys = train_df["text"].apply(
        lambda x: normalise_text(x, normalise_whitespace=normalise_ws)
    )
    val_keys = val_df["text"].apply(
        lambda x: normalise_text(x, normalise_whitespace=normalise_ws)
    )

    if not case_sensitive:
        train_keys = train_keys.str.lower()
        val_keys = val_keys.str.lower()

    val_key_set = set(val_keys.tolist())

    is_validation = train_keys.isin(val_key_set)

    removed_df = train_df[is_validation].copy()
    train_pool_df = train_df[~is_validation].copy()

    return train_pool_df, removed_df


# ---------------------------------------------------------------------
# Fold output and summaries
# ---------------------------------------------------------------------


def save_fold_outputs(
    fold_outdir: Path,
    train_pool_df: pd.DataFrame | None,
    removed_df: pd.DataFrame | None,
    val_df: pd.DataFrame,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> None:
    fold_outdir.mkdir(parents=True, exist_ok=True)

    if train_pool_df is not None and bool(
        get_nested(config, "outputs", "save_train_pool_per_fold", default=True)
    ):
        train_pool_df.to_csv(fold_outdir / "fold_train_pool.csv", index=False)

    if removed_df is not None and bool(
        get_nested(config, "outputs", "save_removed_validation_rows", default=True)
    ):
        removed_df.to_csv(fold_outdir / "removed_validation_rows.csv", index=False)

    val_df.to_csv(fold_outdir / "validation_predictions_standardised.csv", index=False)

    pd.DataFrame([summary]).to_csv(fold_outdir / "fold_summary.csv", index=False)


def make_fold_summary(
    fold_id: int,
    pred_path: Path,
    train_df: pd.DataFrame | None,
    train_pool_df: pd.DataFrame | None,
    removed_df: pd.DataFrame | None,
    val_df: pd.DataFrame,
    available_methods: list[str],
    metric_methods: list[str] | None = None,
) -> dict[str, Any]:
    n_val = len(val_df)

    summary: dict[str, Any] = {
        "fold_id": fold_id,
        "prediction_file": str(pred_path),
        "validation_prediction_n": n_val,
        "validation_accuracy": float(val_df["correct"].mean()) if n_val else np.nan,
        "validation_correct_n": int(val_df["correct"].sum()) if n_val else 0,
        "validation_incorrect_n": int((1 - val_df["correct"]).sum()) if n_val else 0,
        "available_methods_raw": ";".join(available_methods),
        "metric_methods": ";".join(metric_methods or []),
    }

    if train_df is not None:
        n_removed = len(removed_df) if removed_df is not None else 0
        summary.update(
            {
                "original_train_n": len(train_df),
                "removed_from_train_n": n_removed,
                "fold_train_pool_n": len(train_pool_df) if train_pool_df is not None else np.nan,
                "removed_matches_validation_n": bool(n_removed == n_val),
                "train_pool_simple_n": int((train_pool_df["label"] == "simple").sum())
                if train_pool_df is not None and "label" in train_pool_df.columns
                else np.nan,
                "train_pool_complex_n": int((train_pool_df["label"] == "complex").sum())
                if train_pool_df is not None and "label" in train_pool_df.columns
                else np.nan,
            }
        )
    else:
        summary.update(
            {
                "original_train_n": np.nan,
                "removed_from_train_n": np.nan,
                "fold_train_pool_n": np.nan,
                "removed_matches_validation_n": np.nan,
                "train_pool_simple_n": np.nan,
                "train_pool_complex_n": np.nan,
            }
        )

    return summary


def evaluate_available_scores_for_fold(
    config: dict[str, Any],
    fold_id: int,
    fold_outdir: Path,
    val_df: pd.DataFrame,
    available_methods: list[str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, float], list[str]]:
    """Compute metrics for currently available uncertainty scores."""
    if not available_methods:
        print(f"Fold {fold_id}: no available uncertainty methods to evaluate.")
        return None, None, {}, []

    val_df = add_label_indices(config, val_df)

    scores_for_metrics_df, metric_methods = score_columns_for_metrics(
        config=config,
        score_df=val_df,
        methods=available_methods,
    )

    if not metric_methods:
        print(f"Fold {fold_id}: no metric-ready uncertainty methods.")
        return None, None, {}, []

    bins = int(get_nested(config, "metrics", "ece_bins", default=15))
    ti_fixed_coverage = float(
        get_nested(config, "metrics", "ti_fixed_coverage", default=0.95)
    )

    metrics_df, method_times = compute_metrics_per_method_with_timing(
        df=scores_for_metrics_df,
        methods=metric_methods,
        bins=bins,
        ti_fixed_coverage=ti_fixed_coverage,
    )

    metrics_long = metrics_to_long(metrics_df, fold=fold_id)

    fold_outdir.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(fold_outdir / "scores_raw.csv", index=False)
    scores_for_metrics_df.to_csv(fold_outdir / "scores_for_metrics.csv", index=False)
    metrics_df.to_csv(fold_outdir / "metrics_wide.csv")
    metrics_long.to_csv(fold_outdir / "metrics_long.csv", index=False)

    pd.Series(method_times, name="seconds").rename_axis("method").reset_index().to_csv(
        fold_outdir / "metric_times.csv",
        index=False,
    )

    print(f"Fold {fold_id}: evaluated methods: {metric_methods}")
    print(metrics_df.round(4).to_string())

    return metrics_df, metrics_long, method_times, metric_methods


def save_all_fold_outputs(
    outdir: Path,
    summaries: list[dict[str, Any]],
    all_metrics_long: list[pd.DataFrame],
    all_metric_times: list[pd.DataFrame],
) -> None:
    """Save all-fold summary and aggregated metric files."""
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(outdir / "all_folds_summary.csv", index=False)
        # Backward-compatible name used by earlier script.
        summary_df.to_csv(outdir / "part_a_all_folds_summary.csv", index=False)

        print("\nSaved fold summary:")
        print(outdir / "all_folds_summary.csv")
        print(summary_df.to_string(index=False))

    if all_metrics_long:
        metrics_long_all = pd.concat(all_metrics_long, ignore_index=True)
        metrics_long_all.to_csv(outdir / "metrics_long_all_folds.csv", index=False)

        metrics_summary = (
            metrics_long_all.groupby(["method", "metric"], as_index=False)["value"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
        metrics_summary.to_csv(outdir / "metrics_summary_by_method.csv", index=False)

        print("\nSaved all-fold metrics:")
        print(outdir / "metrics_long_all_folds.csv")
        print(outdir / "metrics_summary_by_method.csv")
        print(metrics_summary.round(4).to_string(index=False))

    if all_metric_times:
        times_df = pd.concat(all_metric_times, ignore_index=True)
        times_df.to_csv(outdir / "metric_times_all_folds.csv", index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)
    if config_path is None:
        raise ValueError("Could not resolve --config path.")

    config = load_config(config_path)

    outdir = resolve_path(get_nested(config, "outputs", "outdir"))
    if outdir is None:
        raise ValueError("Missing required config key: outputs.outdir")
    outdir.mkdir(parents=True, exist_ok=True)

    fold_ids = get_nested(config, "folds", "fold_ids", default=[])
    if not fold_ids:
        n_folds = get_nested(config, "folds", "n_folds", default=None)
        if n_folds is not None:
            fold_ids = list(range(int(n_folds)))

    if not fold_ids:
        raise ValueError("No folds provided. Set folds.fold_ids or folds.n_folds in config.")

    skip_missing_prediction_files = bool(
        get_nested(config, "folds", "skip_missing_prediction_files", default=True)
    )

    train_df, _train_text_col, _train_label_col = load_original_train(config)

    all_summaries: list[dict[str, Any]] = []
    all_metrics_long: list[pd.DataFrame] = []
    all_metric_times: list[pd.DataFrame] = []

    for fold_id in fold_ids:
        pred_template = get_nested(config, "data", "pred_csv_template")
        pred_path = resolve_path(format_template(str(pred_template), int(fold_id)))

        if pred_path is None:
            raise ValueError("Could not resolve prediction path.")

        if not pred_path.exists():
            message = f"Fold {fold_id}: missing prediction file: {pred_path}"

            if skip_missing_prediction_files:
                print("Skipping.", message)
                continue

            raise FileNotFoundError(message)

        print(f"\nProcessing fold {fold_id}")
        print(f"Prediction file: {pred_path}")

        val_df, pred_path = load_validation_predictions(config, int(fold_id))
        val_df = add_probability_columns(config, val_df)

        # Deterministic scores derived from saved probabilities.
        val_df, available_methods = add_deterministic_uncertainty_scores(
            config=config,
            val_df=val_df,
        )

        # Already-saved score columns, e.g. SMP from MC dropout.
        val_df, available_methods = add_existing_uncertainty_score_columns(
            config=config,
            val_df=val_df,
            available_methods=available_methods,
        )

        train_pool_df, removed_df = reconstruct_train_pool(
            config=config,
            train_df=train_df,
            val_df=val_df,
        )

        fold_outdir = outdir / f"fold_{fold_id}"

        metrics_df, metrics_long, method_times, metric_methods = evaluate_available_scores_for_fold(
            config=config,
            fold_id=int(fold_id),
            fold_outdir=fold_outdir,
            val_df=val_df,
            available_methods=available_methods,
        )

        summary = make_fold_summary(
            fold_id=int(fold_id),
            pred_path=pred_path,
            train_df=train_df,
            train_pool_df=train_pool_df,
            removed_df=removed_df,
            val_df=val_df,
            available_methods=available_methods,
            metric_methods=metric_methods,
        )

        save_fold_outputs(
            fold_outdir=fold_outdir,
            train_pool_df=train_pool_df,
            removed_df=removed_df,
            val_df=val_df,
            summary=summary,
            config=config,
        )

        all_summaries.append(summary)

        if metrics_long is not None:
            all_metrics_long.append(metrics_long)

        if method_times:
            times_df = pd.Series(method_times, name="seconds").rename_axis("method").reset_index()
            times_df["fold"] = int(fold_id)
            all_metric_times.append(times_df)

        print(pd.DataFrame([summary]).to_string(index=False))

    if all_summaries:
        save_all_fold_outputs(
            outdir=outdir,
            summaries=all_summaries,
            all_metrics_long=all_metrics_long,
            all_metric_times=all_metric_times,
        )
    else:
        print("No folds were processed.")


if __name__ == "__main__":
    main()
