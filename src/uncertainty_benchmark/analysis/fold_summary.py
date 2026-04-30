#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uncertainty_benchmark.analysis.fold_summary

Utilities for loading, cleaning, stacking, and summarising fold-level metric
summary files.

This module contains reusable logic behind scripts such as:

    scripts/summarize_fold_metrics.py

Expected input format
---------------------
Each fold-level CSV should have:

    rows    = metric names
    columns = uncertainty methods
    values  = metric values

Example:

    Metric,SR,SMP,ENT,ENT_MC,PV,BALD
    ROC-AUC,0.71,0.69,0.73,0.74,0.70,0.72
    AU-PRC,0.42,0.40,0.45,0.46,0.41,0.43
    ECE,0.12,0.14,0.11,0.10,0.13,0.12

Main features
-------------
- metric alias normalisation
- loading fold-level metric CSVs
- selecting common methods and metrics across folds
- stacking fold summaries into a MultiIndex DataFrame
- computing mean and standard deviation tables
- choosing best methods using metric-specific rules
- producing tidy long-form summaries
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------

HIGHER_IS_BETTER = {
    "ROC-AUC",
    "AU-PRC",
    "AU-PRC (E)",
    "AU-PRC (C)",
    "Norm RC-AUC",
    "N.RC-AUC",
    "NRC-AUC",
    "TI",
    "TI@95",
}

TARGET_VALUE_METRICS = {
    "C-Slope": 1.0,
    "CITL": 0.0,
}

METRIC_ALIASES = {
    "roc-auc": "ROC-AUC",
    "au-prc": "AU-PRC",
    "au-prc (e)": "AU-PRC (E)",
    "au-prc (c)": "AU-PRC (C)",
    "n.rc-auc": "Norm RC-AUC",
    "nrc-auc": "Norm RC-AUC",
    "norm rc-auc": "Norm RC-AUC",
    "e-auopt rc": "E-AUoptRC",
    "e-auopt": "E-AUoptRC",
    "e-auoptrc": "E-AUoptRC",
    "calibration slope": "C-Slope",
    "slope": "C-Slope",
    "c-slope": "C-Slope",
    "cal-in-the-large": "CITL",
    "calibration-in-the-large": "CITL",
    "citl": "CITL",
    "expected calibration error": "ECE",
    "ece": "ECE",
    "ti-95": "TI@95",
    "ti @95": "TI@95",
}

METRIC_GROUPS = {
    "Uncertainty Discrimination": [
        "ROC-AUC",
        "AU-PRC",
        "AU-PRC (E)",
        "AU-PRC (C)",
    ],
    "Calibration Metrics": [
        "C-Slope",
        "CITL",
        "ECE",
    ],
    "Selective Prediction Metrics": [
        "RC-AUC",
        "Norm RC-AUC",
        "E-AUoptRC",
        "TI",
        "TI@95",
    ],
}

DEFAULT_METHOD_ORDER = [
    "SR",
    "SMP",
    "ENT",
    "ENT_MC",
    "PV",
    "BALD",
    "MD",
    "HUQ-MD",
    "LOF",
    "ISOF",
]

METHOD_DISPLAY_NAMES = {
    "ENT_MC": "ENT-MC",
}

METRIC_DISPLAY_NAMES = {
    "Norm RC-AUC": "N.RC-AUC",
}


# ---------------------------------------------------------------------
# Metric-name and method helpers
# ---------------------------------------------------------------------

def normalise_metric_name(name: object) -> str:
    """Normalise metric aliases while preserving readable names.

    Parameters
    ----------
    name:
        Raw metric name from a CSV index.

    Returns
    -------
    str
        Canonical metric name when known; otherwise the stripped original name.
    """
    raw = str(name).strip()
    key = raw.lower()
    return METRIC_ALIASES.get(key, raw)


def normalise_metric_index(index: Iterable[object]) -> List[str]:
    """Normalise all metric names in an index-like object."""
    return [normalise_metric_name(x) for x in index]


def ordered_methods(methods: Iterable[str], preferred_order: Sequence[str] = DEFAULT_METHOD_ORDER) -> List[str]:
    """Order methods using a preferred list, then append any remaining methods."""
    method_list = list(methods)
    ordered = [m for m in preferred_order if m in method_list]
    ordered.extend([m for m in method_list if m not in ordered])
    return ordered


def group_metrics(
    metrics: Iterable[str],
    metric_groups: Mapping[str, Sequence[str]] = METRIC_GROUPS,
) -> List[Tuple[str, List[str]]]:
    """Group metrics into known groups plus an 'Other Metrics' group."""
    available = list(metrics)
    seen = set()
    groups: List[Tuple[str, List[str]]] = []

    for group_name, group_metric_order in metric_groups.items():
        selected = [metric for metric in group_metric_order if metric in available]
        if selected:
            groups.append((group_name, selected))
            seen.update(selected)

    other = [metric for metric in available if metric not in seen]
    if other:
        groups.append(("Other Metrics", other))

    return groups


# ---------------------------------------------------------------------
# Loading fold summaries
# ---------------------------------------------------------------------

def load_fold_summary(
    path: str | Path,
    index_col: int | str = 0,
    normalise_metrics: bool = True,
    drop_duplicate_metrics: bool = True,
) -> pd.DataFrame:
    """Load one fold-level metric summary CSV.

    Parameters
    ----------
    path:
        CSV file path.
    index_col:
        Column used as metric-name index. Default: first column.
    normalise_metrics:
        Whether to normalise metric aliases.
    drop_duplicate_metrics:
        Whether to keep only the first duplicate metric row.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows=metrics and columns=methods.
    """
    df = pd.read_csv(path, index_col=index_col)

    if normalise_metrics:
        df.index = normalise_metric_index(df.index)

    if drop_duplicate_metrics:
        df = df[~pd.Index(df.index).duplicated(keep="first")]

    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_fold_summaries(
    summary_glob: str,
    index_col: int | str = 0,
    normalise_metrics: bool = True,
    drop_duplicate_metrics: bool = True,
) -> Tuple[List[Path], List[pd.DataFrame]]:
    """Load all fold summary files matching a glob pattern."""
    files = sorted(Path(path) for path in glob.glob(summary_glob))
    if not files:
        raise FileNotFoundError(f"No fold summary files matched: {summary_glob}")

    fold_dfs = [
        load_fold_summary(
            path,
            index_col=index_col,
            normalise_metrics=normalise_metrics,
            drop_duplicate_metrics=drop_duplicate_metrics,
        )
        for path in files
    ]
    return files, fold_dfs


# ---------------------------------------------------------------------
# Common methods/metrics and stacking
# ---------------------------------------------------------------------

def common_methods_across_folds(fold_dfs: Sequence[pd.DataFrame]) -> List[str]:
    """Return methods present in every fold, preserving first-fold order."""
    if not fold_dfs:
        return []

    first_methods = list(fold_dfs[0].columns)
    return [method for method in first_methods if all(method in df.columns for df in fold_dfs)]


def common_metrics_across_folds(fold_dfs: Sequence[pd.DataFrame]) -> List[str]:
    """Return metrics present in every fold, preserving first-fold order."""
    if not fold_dfs:
        return []

    first_metrics = list(fold_dfs[0].index)
    return [metric for metric in first_metrics if all(metric in df.index for df in fold_dfs)]


def union_methods_across_folds(fold_dfs: Sequence[pd.DataFrame]) -> List[str]:
    """Return union of methods across folds, preserving first occurrence order."""
    seen = set()
    methods: List[str] = []
    for df in fold_dfs:
        for method in df.columns:
            if method not in seen:
                seen.add(method)
                methods.append(method)
    return methods


def union_metrics_across_folds(fold_dfs: Sequence[pd.DataFrame]) -> List[str]:
    """Return union of metrics across folds, preserving first occurrence order."""
    seen = set()
    metrics: List[str] = []
    for df in fold_dfs:
        for metric in df.index:
            if metric not in seen:
                seen.add(metric)
                metrics.append(metric)
    return metrics


def stack_fold_summaries(
    fold_dfs: Sequence[pd.DataFrame],
    use_common_methods: bool = True,
    use_common_metrics: bool = True,
) -> pd.DataFrame:
    """Stack fold summary DataFrames into one MultiIndex DataFrame.

    Parameters
    ----------
    fold_dfs:
        List of fold-level DataFrames.
    use_common_methods:
        If True, keep only methods present in every fold. If False, use the
        union and allow NaNs for missing method values.
    use_common_metrics:
        If True, keep only metrics present in every fold. If False, use the
        union and allow NaNs for missing metric values.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with index=(fold, metric) and columns=methods.
    """
    if not fold_dfs:
        raise ValueError("No fold DataFrames were provided.")

    methods = common_methods_across_folds(fold_dfs) if use_common_methods else union_methods_across_folds(fold_dfs)
    metrics = common_metrics_across_folds(fold_dfs) if use_common_metrics else union_metrics_across_folds(fold_dfs)

    if len(methods) < 1:
        raise ValueError("No usable methods found across folds.")
    if len(metrics) < 1:
        raise ValueError("No usable metrics found across folds.")

    aligned: List[pd.DataFrame] = []
    for df in fold_dfs:
        aligned_df = df.reindex(index=metrics, columns=methods).copy()
        aligned.append(aligned_df)

    return pd.concat(aligned, keys=range(len(aligned)), names=["fold", "metric"])


def load_and_stack_fold_summaries(
    summary_glob: str,
    use_common_methods: bool = True,
    use_common_metrics: bool = True,
) -> Tuple[List[Path], pd.DataFrame]:
    """Convenience wrapper for loading and stacking fold summary files."""
    files, fold_dfs = load_fold_summaries(summary_glob)
    stacked = stack_fold_summaries(
        fold_dfs,
        use_common_methods=use_common_methods,
        use_common_metrics=use_common_metrics,
    )
    return files, stacked


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def mean_std_tables(stacked: pd.DataFrame, metric_level: str = "metric") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute mean and standard deviation tables across folds."""
    if not isinstance(stacked.index, pd.MultiIndex):
        raise ValueError("stacked must have a MultiIndex, usually (fold, metric).")

    if metric_level not in stacked.index.names:
        raise ValueError(f"Metric level {metric_level!r} not found in index names {stacked.index.names}.")

    mean_values = stacked.groupby(level=metric_level).mean().astype(float)
    std_values = stacked.groupby(level=metric_level).std().astype(float)
    return mean_values, std_values


def summary_mean_std_frame(stacked: pd.DataFrame, metric_level: str = "metric") -> pd.DataFrame:
    """Return a wide summary DataFrame with top-level columns mean/std."""
    mean_values, std_values = mean_std_tables(stacked, metric_level=metric_level)
    return pd.concat({"mean": mean_values, "std": std_values}, axis=1)


def tidy_mean_std_summary(
    stacked: pd.DataFrame,
    metric_level: str = "metric",
) -> pd.DataFrame:
    """Return a tidy summary with one row per metric-method pair."""
    mean_values, std_values = mean_std_tables(stacked, metric_level=metric_level)

    rows: List[Dict[str, object]] = []
    for metric in mean_values.index:
        for method in mean_values.columns:
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "mean": mean_values.loc[metric, method],
                    "std": std_values.loc[metric, method],
                }
            )

    return pd.DataFrame(rows)


def fold_count_by_metric(stacked: pd.DataFrame, metric_level: str = "metric") -> pd.Series:
    """Return number of non-missing folds per metric/method combination."""
    if not isinstance(stacked.index, pd.MultiIndex):
        raise ValueError("stacked must have a MultiIndex, usually (fold, metric).")
    return stacked.groupby(level=metric_level).count()


# ---------------------------------------------------------------------
# Best-method selection
# ---------------------------------------------------------------------

def choose_best_method_for_metric(
    metric: str,
    mean_row: pd.Series,
    higher_is_better: Iterable[str] = HIGHER_IS_BETTER,
    target_value_metrics: Mapping[str, float] = TARGET_VALUE_METRICS,
) -> str:
    """Choose the best method for one metric.

    Rules:
      - higher-is-better metrics: choose max
      - target-value metrics: choose value closest to target
      - all other metrics: choose min
    """
    values = pd.Series(mean_row, dtype=float).dropna()
    if values.empty:
        return ""

    if metric in target_value_metrics:
        target = float(target_value_metrics[metric])
        return str((values - target).abs().idxmin())

    if metric in set(higher_is_better):
        return str(values.idxmax())

    return str(values.idxmin())


def choose_best_methods(
    mean_values: pd.DataFrame,
    higher_is_better: Iterable[str] = HIGHER_IS_BETTER,
    target_value_metrics: Mapping[str, float] = TARGET_VALUE_METRICS,
) -> Dict[str, str]:
    """Choose the best method for every metric in a mean-value table."""
    best: Dict[str, str] = {}
    for metric in mean_values.index:
        best[str(metric)] = choose_best_method_for_metric(
            str(metric),
            mean_values.loc[metric],
            higher_is_better=higher_is_better,
            target_value_metrics=target_value_metrics,
        )
    return best


def add_best_method_column(
    tidy_summary: pd.DataFrame,
    best_methods: Mapping[str, str],
) -> pd.DataFrame:
    """Add boolean column indicating whether a row is the best for its metric."""
    df = tidy_summary.copy()
    df["is_best"] = [row["method"] == best_methods.get(row["metric"], "") for _, row in df.iterrows()]
    return df


# ---------------------------------------------------------------------
# Fold-level diagnostics
# ---------------------------------------------------------------------

def describe_loaded_folds(files: Sequence[Path], fold_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Return diagnostics about loaded fold summary files."""
    rows: List[Dict[str, object]] = []
    for fold_id, (path, df) in enumerate(zip(files, fold_dfs)):
        rows.append(
            {
                "fold": fold_id,
                "file": str(path),
                "n_metrics": int(df.shape[0]),
                "n_methods": int(df.shape[1]),
                "metrics": ";".join(map(str, df.index.tolist())),
                "methods": ";".join(map(str, df.columns.tolist())),
            }
        )
    return pd.DataFrame(rows)


def missingness_summary(stacked: pd.DataFrame, metric_level: str = "metric") -> pd.DataFrame:
    """Summarise missing values by metric and method."""
    if not isinstance(stacked.index, pd.MultiIndex):
        raise ValueError("stacked must have a MultiIndex, usually (fold, metric).")

    rows: List[Dict[str, object]] = []
    for metric, sub in stacked.groupby(level=metric_level):
        for method in sub.columns:
            values = sub[method]
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "n_total": int(values.shape[0]),
                    "n_missing": int(values.isna().sum()),
                    "n_observed": int(values.notna().sum()),
                    "pct_missing": float(100.0 * values.isna().mean()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------

def summarise_fold_metrics(
    summary_glob: str,
    use_common_methods: bool = True,
    use_common_metrics: bool = True,
) -> Dict[str, object]:
    """Load fold summaries and compute standard aggregate outputs.

    Returns a dictionary containing:
      - files
      - stacked
      - mean
      - std
      - summary
      - tidy_summary
      - best_methods
    """
    files, fold_dfs = load_fold_summaries(summary_glob)
    stacked = stack_fold_summaries(
        fold_dfs,
        use_common_methods=use_common_methods,
        use_common_metrics=use_common_metrics,
    )
    mean_values, std_values = mean_std_tables(stacked)
    summary = pd.concat({"mean": mean_values, "std": std_values}, axis=1)
    tidy = tidy_mean_std_summary(stacked)
    best = choose_best_methods(mean_values)
    tidy = add_best_method_column(tidy, best)

    return {
        "files": files,
        "fold_dfs": fold_dfs,
        "stacked": stacked,
        "mean": mean_values,
        "std": std_values,
        "summary": summary,
        "tidy_summary": tidy,
        "best_methods": best,
        "loaded_folds": describe_loaded_folds(files, fold_dfs),
        "missingness": missingness_summary(stacked),
    }


__all__ = [
    "HIGHER_IS_BETTER",
    "TARGET_VALUE_METRICS",
    "METRIC_ALIASES",
    "METRIC_GROUPS",
    "DEFAULT_METHOD_ORDER",
    "METHOD_DISPLAY_NAMES",
    "METRIC_DISPLAY_NAMES",
    "normalise_metric_name",
    "normalise_metric_index",
    "ordered_methods",
    "group_metrics",
    "load_fold_summary",
    "load_fold_summaries",
    "common_methods_across_folds",
    "common_metrics_across_folds",
    "union_methods_across_folds",
    "union_metrics_across_folds",
    "stack_fold_summaries",
    "load_and_stack_fold_summaries",
    "mean_std_tables",
    "summary_mean_std_frame",
    "tidy_mean_std_summary",
    "fold_count_by_metric",
    "choose_best_method_for_metric",
    "choose_best_methods",
    "add_best_method_column",
    "describe_loaded_folds",
    "missingness_summary",
    "summarise_fold_metrics",
]
