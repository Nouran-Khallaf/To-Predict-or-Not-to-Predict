#!/usr/bin/env python3
"""
Fit and apply SMP rejection thresholds.

Main idea:
- Use validation predictions to find the SMP cutoff.
- Save the cutoff.
- Apply the saved cutoff to new/unseen predictions.

Convention:
- SMP is treated as an uncertainty score.
- larger SMP = more uncertain
- accept if SMP <= threshold
- reject if SMP > threshold
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
import copy
import glob
import re

# -----------------------------
# Basic helpers
# -----------------------------

def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mkdir_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def get_score_direction(config: Dict) -> str:
    return config.get("uncertainty", {}).get(
        "score_direction",
        "higher_is_uncertain"
    )


def sort_most_confident_first(
    df: pd.DataFrame,
    uncertainty_col: str,
    score_direction: str,
) -> pd.DataFrame:
    if score_direction == "higher_is_uncertain":
        return df.sort_values(uncertainty_col, ascending=True)

    if score_direction == "lower_is_uncertain":
        return df.sort_values(uncertainty_col, ascending=False)

    raise ValueError(
        "score_direction must be 'higher_is_uncertain' or 'lower_is_uncertain'"
    )


def get_threshold_from_accepted(
    accepted: pd.DataFrame,
    uncertainty_col: str,
    score_direction: str,
) -> float:
    if score_direction == "higher_is_uncertain":
        return float(accepted[uncertainty_col].max())

    return float(accepted[uncertainty_col].min())


def apply_threshold_rule(
    scores: pd.Series,
    threshold,
    score_direction: str,
) -> pd.Series:
    """
    Returns reject=True/False.

    For SMP:
        higher_is_uncertain
        reject if SMP > threshold
    """

    if score_direction == "higher_is_uncertain":
        return scores > threshold

    return scores < threshold


# -----------------------------
# Coverage grid
# -----------------------------

def build_coverage_grid(config: Dict) -> List[float]:
    grid_cfg = config.get("coverage_grid", {})

    start = float(grid_cfg.get("start", 1.00))
    end = float(grid_cfg.get("end", 0.50))
    step = float(grid_cfg.get("step", 0.05))

    coverages = list(np.arange(start, end - 1e-9, -abs(step)))

    selection_cfg = config.get("threshold_selection", {})
    if selection_cfg.get("mode") == "target_coverage":
        coverages.append(float(selection_cfg.get("target_coverage", 0.80)))

    coverages = sorted(set(round(x, 4) for x in coverages), reverse=True)
    return coverages


# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(
    accepted: pd.DataFrame,
    config: Dict,
) -> Dict[str, float]:
    task_type = config.get("task_type", "classification")
    cols = config["columns"]

    gold_col = cols.get("gold_col")
    pred_col = cols.get("pred_col")

    if gold_col is None or pred_col is None:
        return {}

    if task_type == "classification":
        accuracy = float((accepted[gold_col] == accepted[pred_col]).mean())
        return {
            "accepted_accuracy": accuracy,
            "accepted_error_rate": 1.0 - accuracy,
        }

    if task_type == "regression":
        return {
            "accepted_rmse": rmse(accepted[gold_col], accepted[pred_col]),
            "accepted_mae": mae(accepted[gold_col], accepted[pred_col]),
        }

    raise ValueError("task_type must be 'classification' or 'regression'")


# -----------------------------
# Rejection curve
# -----------------------------

def rejection_curve_for_group(
    df: pd.DataFrame,
    group_name: str,
    config: Dict,
) -> pd.DataFrame:
    cols = config["columns"]

    uncertainty_col = cols["uncertainty_col"]
    method_name = config.get("uncertainty", {}).get("method_name", uncertainty_col)
    score_direction = get_score_direction(config)

    required = [uncertainty_col]

    gold_col = cols.get("gold_col")
    pred_col = cols.get("pred_col")

    if gold_col:
        required.append(gold_col)
    if pred_col:
        required.append(pred_col)

    df = df.dropna(subset=required).copy()

    if len(df) == 0:
        raise ValueError(f"No usable rows for group: {group_name}")

    df = sort_most_confident_first(
        df,
        uncertainty_col=uncertainty_col,
        score_direction=score_direction,
    ).reset_index(drop=True)

    n = len(df)
    rows = []

    for requested_coverage in build_coverage_grid(config):
        k = int(np.floor(requested_coverage * n))
        k = max(1, min(k, n))

        accepted = df.iloc[:k].copy()
        rejected = df.iloc[k:].copy()

        threshold = get_threshold_from_accepted(
            accepted,
            uncertainty_col=uncertainty_col,
            score_direction=score_direction,
        )

        row = {
            "language": group_name,
            "method": method_name,
            "score_direction": score_direction,
            "requested_coverage": requested_coverage,
            "actual_coverage": len(accepted) / n,
            "rejection_rate": len(rejected) / n,
            "threshold": threshold,
            "accepted_n": len(accepted),
            "rejected_n": len(rejected),
            "total_n": n,
        }

        row.update(compute_metrics(accepted, config))
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Threshold selection
# -----------------------------

def select_threshold_row(curve: pd.DataFrame, config: Dict) -> pd.Series:
    selection = config.get("threshold_selection", {})
    mode = selection.get("mode", "target_coverage")

    curve = curve.copy()

    if mode == "target_coverage":
        target = float(selection.get("target_coverage", 0.80))
        curve["_distance"] = (curve["actual_coverage"] - target).abs()
        return curve.sort_values(
            ["_distance", "actual_coverage"],
            ascending=[True, False],
        ).iloc[0]

    if mode == "target_accuracy":
        target = float(selection["target_accuracy"])
        valid = curve[curve["accepted_accuracy"] >= target]
        if len(valid) == 0:
            raise ValueError(f"No threshold reached accuracy >= {target}")
        return valid.sort_values("actual_coverage", ascending=False).iloc[0]

    if mode == "target_error_rate":
        target = float(selection["target_error_rate"])
        valid = curve[curve["accepted_error_rate"] <= target]
        if len(valid) == 0:
            raise ValueError(f"No threshold reached error rate <= {target}")
        return valid.sort_values("actual_coverage", ascending=False).iloc[0]

    if mode == "target_rmse":
        target = float(selection["target_rmse"])
        valid = curve[curve["accepted_rmse"] <= target]
        if len(valid) == 0:
            raise ValueError(f"No threshold reached RMSE <= {target}")
        return valid.sort_values("actual_coverage", ascending=False).iloc[0]

    if mode == "target_mae":
        target = float(selection["target_mae"])
        valid = curve[curve["accepted_mae"] <= target]
        if len(valid) == 0:
            raise ValueError(f"No threshold reached MAE <= {target}")
        return valid.sort_values("actual_coverage", ascending=False).iloc[0]

    raise ValueError(
        "threshold_selection.mode must be one of: "
        "target_coverage, target_accuracy, target_error_rate, target_rmse, target_mae"
    )


# -----------------------------
# Bootstrap threshold stability
# -----------------------------

def bootstrap_threshold(
    df: pd.DataFrame,
    group_name: str,
    config: Dict,
) -> Dict[str, float]:
    bootstrap_cfg = config.get("bootstrap", {})
    enabled = bool(bootstrap_cfg.get("enabled", False))

    if not enabled:
        return {}

    n_resamples = int(bootstrap_cfg.get("n_resamples", 200))
    random_seed = int(bootstrap_cfg.get("random_seed", 42))
    ci = float(bootstrap_cfg.get("ci", 0.95))

    rng = np.random.default_rng(random_seed)

    thresholds = []

    for _ in range(n_resamples):
        sample_idx = rng.integers(0, len(df), size=len(df))
        sample = df.iloc[sample_idx].copy()

        curve = rejection_curve_for_group(
            sample,
            group_name=group_name,
            config=config,
        )

        chosen = select_threshold_row(curve, config)
        thresholds.append(float(chosen["threshold"]))

    alpha = 1.0 - ci
    lower = np.quantile(thresholds, alpha / 2)
    upper = np.quantile(thresholds, 1 - alpha / 2)

    return {
        "threshold_boot_mean": float(np.mean(thresholds)),
        "threshold_boot_std": float(np.std(thresholds)),
        "threshold_boot_ci_lower": float(lower),
        "threshold_boot_ci_upper": float(upper),
        "threshold_boot_n": n_resamples,
    }


# -----------------------------
# Language filtering
# -----------------------------

def filter_languages(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    lang_cfg = config.get("languages", {})
    use = lang_cfg.get("use", "all")

    if use == "all":
        return df

    if use != "selected":
        raise ValueError("languages.use must be 'all' or 'selected'")

    language_col = config["columns"].get("language_col")
    selected = lang_cfg.get("selected", [])

    if language_col not in df.columns:
        raise ValueError(f"Language column not found: {language_col}")

    selected = [str(x) for x in selected]

    df = df.copy()
    df[language_col] = df[language_col].astype(str)

    return df[df[language_col].isin(selected)].copy()


# -----------------------------
# Fit thresholds
# -----------------------------
def load_fit_predictions(config: Dict) -> pd.DataFrame:
    """
    Load validation prediction files.

    Supports either:
    - input_predictions: one CSV file
    - input_predictions_glob: many fold CSV files
    """

    if "input_predictions_glob" in config:
        pattern = config["input_predictions_glob"]
        paths = sorted(glob.glob(pattern))

        if not paths:
            raise FileNotFoundError(
                f"No files matched input_predictions_glob: {pattern}"
            )

        dfs = []

        for path in paths:
            fold_df = pd.read_csv(path)

            match = re.search(r"fold_(\d+)", path)
            if match:
                fold_df["fold"] = int(match.group(1))

            fold_df["source_file"] = path
            dfs.append(fold_df)

        return pd.concat(dfs, ignore_index=True)

    if "input_predictions" in config:
        return pd.read_csv(config["input_predictions"])

    raise KeyError(
        "Config must contain either 'input_predictions' or 'input_predictions_glob'."
    )


def get_uncertainty_methods(config: Dict) -> List[str]:
    """
    Allows config to use either:

    uncertainty:
      methods: ["SMP", "SR"]

    or old style:

    columns:
      uncertainty_col: "SMP"
    """

    methods = config.get("uncertainty", {}).get("methods")

    if methods:
        return [str(m) for m in methods]

    uncertainty_col = config.get("columns", {}).get("uncertainty_col")

    if uncertainty_col:
        return [str(uncertainty_col)]

    raise ValueError(
        "No uncertainty method found. Add uncertainty.methods or columns.uncertainty_col."
    )

def fit_thresholds(config: Dict) -> None:
    output_thresholds = config["output_thresholds"]
    output_curve = config.get("output_curve")

    df = load_fit_predictions(config)
    df = filter_languages(df, config)

    threshold_scope = config.get("threshold", {}).get("scope", "by_language")
    language_col = config["columns"].get("language_col")

    methods = get_uncertainty_methods(config)

    all_curves = []
    chosen_rows = []

    for method in methods:
        if method not in df.columns:
            raise ValueError(
                f"Uncertainty method column not found in prediction files: {method}\n"
                f"Available columns are: {list(df.columns)}"
            )

        method_config = copy.deepcopy(config)
        method_config["columns"]["uncertainty_col"] = method
        method_config.setdefault("uncertainty", {})
        method_config["uncertainty"]["method_name"] = method

        if threshold_scope == "global":
            curve = rejection_curve_for_group(df, "ALL", method_config)
            chosen = select_threshold_row(curve, method_config)

            boot = bootstrap_threshold(df, "ALL", method_config)

            chosen_dict = chosen.to_dict()
            chosen_dict.update(boot)

            all_curves.append(curve)
            chosen_rows.append(chosen_dict)

        elif threshold_scope == "by_language":
            if not language_col:
                raise ValueError(
                    "threshold.scope='by_language' requires columns.language_col."
                )

            if language_col not in df.columns:
                raise ValueError(f"Language column not found: {language_col}")

            for lang, group in df.groupby(language_col):
                lang = str(lang)

                curve = rejection_curve_for_group(group, lang, method_config)
                chosen = select_threshold_row(curve, method_config)

                boot = bootstrap_threshold(group, lang, method_config)

                chosen_dict = chosen.to_dict()
                chosen_dict.update(boot)

                all_curves.append(curve)
                chosen_rows.append(chosen_dict)

        else:
            raise ValueError("threshold.scope must be 'global' or 'by_language'")

    threshold_df = pd.DataFrame(chosen_rows)
    threshold_df = threshold_df.drop(columns=["_distance"], errors="ignore")

    mkdir_parent(output_thresholds)
    threshold_df.to_csv(output_thresholds, index=False)

    print(f"Saved thresholds to: {output_thresholds}")
    print(threshold_df)

    if output_curve:
        curve_df = pd.concat(all_curves, ignore_index=True)
        mkdir_parent(output_curve)
        curve_df.to_csv(output_curve, index=False)
        print(f"Saved rejection curve to: {output_curve}")

# -----------------------------
# Apply thresholds
# -----------------------------

def apply_thresholds(config: Dict) -> None:
    input_path = config["input_predictions"]
    threshold_file = config["threshold_file"]
    output_path = config["output_predictions"]

    df = pd.read_csv(input_path)
    thresholds = pd.read_csv(threshold_file)

    cols = config["columns"]
    uncertainty_col = cols["uncertainty_col"]
    language_col = cols.get("language_col")

    method_name = config.get("uncertainty", {}).get("method_name", uncertainty_col)
    score_direction = get_score_direction(config)

    threshold_scope = config.get("threshold", {}).get("scope", "by_language")

    threshold_col = f"{method_name}_threshold"
    reject_col = f"reject_{method_name}"
    accepted_col = f"accepted_{method_name}"

    if uncertainty_col not in df.columns:
        raise ValueError(f"Uncertainty column not found: {uncertainty_col}")

    if threshold_scope == "global":
        if len(thresholds) != 1:
            raise ValueError("Global threshold file should contain exactly one row")

        threshold = float(thresholds.iloc[0]["threshold"])

        df[threshold_col] = threshold
        df[reject_col] = apply_threshold_rule(
            df[uncertainty_col],
            threshold,
            score_direction,
        )
        df[accepted_col] = ~df[reject_col]

    elif threshold_scope == "by_language":
        if language_col not in df.columns:
            raise ValueError(f"Language column not found: {language_col}")

        threshold_map = dict(
            zip(thresholds["language"].astype(str), thresholds["threshold"])
        )

        df[language_col] = df[language_col].astype(str)
        df[threshold_col] = df[language_col].map(threshold_map)

        missing = sorted(df.loc[df[threshold_col].isna(), language_col].unique())

        if missing:
            raise ValueError(
                f"No saved SMP threshold found for these languages: {missing}"
            )

        df[reject_col] = apply_threshold_rule(
            df[uncertainty_col],
            df[threshold_col],
            score_direction,
        )
        df[accepted_col] = ~df[reject_col]

    else:
        raise ValueError("threshold.scope must be 'global' or 'by_language'")

    mkdir_parent(output_path)
    df.to_csv(output_path, index=False)

    print(f"Saved predictions with SMP rejection labels to: {output_path}")
    print(df[[uncertainty_col, threshold_col, reject_col, accepted_col]].head())


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    mode = config.get("mode")

    if mode == "fit_thresholds":
        fit_thresholds(config)

    elif mode == "apply_thresholds":
        apply_thresholds(config)

    else:
        raise ValueError(
            "Config mode must be 'fit_thresholds' or 'apply_thresholds'"
        )


if __name__ == "__main__":
    main()