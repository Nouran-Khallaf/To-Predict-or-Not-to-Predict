#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uncertainty_benchmark.analysis.metric_correlations

Reusable utilities for analysing correlations between uncertainty metrics.

This module supports scripts such as:

    scripts/analyze_metric_correlations.py

The main idea is to compare metric pairs by concatenating method-level metric
values across folds. For example, for one language and one metric pair:

    metric A vector = [fold0/SR, fold0/SMP, ..., fold1/SR, fold1/SMP, ...]
    metric B vector = [fold0/SR, fold0/SMP, ..., fold1/SR, fold1/SMP, ...]

Then compute Kendall's tau and Pearson's r between those two vectors.

Expected input format
---------------------
Each fold-level metric summary CSV should have:

    rows    = metrics
    columns = uncertainty methods
    values  = metric values

Main outputs
------------
- pairwise Kendall tau tables
- pairwise Kendall p-value tables
- pairwise Pearson r tables
- pairwise n tables
- per-language square correlation matrices
"""

from __future__ import annotations

import glob
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr


# ---------------------------------------------------------------------
# Metric aliases and curated metric-pair groups
# ---------------------------------------------------------------------

METRIC_ALIASES: Dict[str, List[str]] = {
    "AU-PRC": ["AU-PRC", "AU-PRC (E)", "AU-PRC (C)"],
    "ROC-AUC": ["ROC-AUC", "roc-auc"],
    "Norm RC-AUC": ["Norm RC-AUC", "N.RC-AUC", "NRC-AUC"],
    "RC-AUC": ["RC-AUC"],
    "E-AUoptRC": ["E-AUoptRC", "E-AUopt RC", "E-AUopt"],
    "TI": ["TI"],
    "TI@95": ["TI@95", "TI-95", "TI @95"],
    "C-Slope": ["C-Slope", "Calibration Slope", "Slope"],
    "CITL": ["CITL", "Cal-in-the-Large", "Calibration-in-the-large"],
    "ECE": ["ECE", "Expected Calibration Error"],
}

CURATED_METRIC_PAIRS_BY_GROUP: Dict[str, List[Tuple[str, str]]] = {
    "Uncertainty Discrimination": [
        ("AU-PRC", "ROC-AUC"),
    ],
    "Calibration Metrics": [
        ("CITL", "ECE"),
        ("C-Slope", "CITL"),
        ("C-Slope", "ECE"),
    ],
    "Selective Prediction Metrics": [
        ("E-AUoptRC", "RC-AUC"),
        ("E-AUoptRC", "Norm RC-AUC"),
        ("E-AUoptRC", "TI"),
        ("Norm RC-AUC", "TI"),
        ("RC-AUC", "TI"),
        ("RC-AUC", "Norm RC-AUC"),
        ("E-AUoptRC", "TI@95"),
        ("Norm RC-AUC", "TI@95"),
        ("RC-AUC", "TI@95"),
    ],
}


# ---------------------------------------------------------------------
# Loading and name-resolution helpers
# ---------------------------------------------------------------------

def first_present(canonical_metric: str, index_like: Iterable[object]) -> Optional[str]:
    """Find the first matching row name for a canonical metric using aliases."""
    aliases = METRIC_ALIASES.get(canonical_metric, [canonical_metric])
    lower_to_original = {str(x).lower(): str(x) for x in index_like}

    for alias in aliases:
        key = alias.lower()
        if key in lower_to_original:
            return lower_to_original[key]

    return None


def load_language_folds(glob_pattern: str) -> List[pd.DataFrame]:
    """Load all fold summary files for one language.

    Returns a list of DataFrames where rows are metrics and columns are methods.
    """
    files = sorted(glob.glob(glob_pattern))
    if not files:
        return []

    fold_dfs: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df[~df.index.duplicated(keep="first")]
        fold_dfs.append(df)

    return fold_dfs


def load_all_languages(lang_globs: Mapping[str, str]) -> Dict[str, List[pd.DataFrame]]:
    """Load fold summary files for multiple languages.

    Parameters
    ----------
    lang_globs:
        Mapping from language code/name to glob pattern.
    """
    return {lang: load_language_folds(pattern) for lang, pattern in lang_globs.items()}


def common_methods_across_folds(fold_dfs: Sequence[pd.DataFrame]) -> List[str]:
    """Return methods present in every fold, preserving first-fold order."""
    if not fold_dfs:
        return []

    first_methods = list(fold_dfs[0].columns)
    return [method for method in first_methods if all(method in df.columns for df in fold_dfs)]


def metric_present_in_all_folds(fold_dfs: Sequence[pd.DataFrame], canonical_metric: str) -> bool:
    """Return True if a canonical metric is present in every fold via aliases."""
    if not fold_dfs:
        return False

    for df in fold_dfs:
        row_name = first_present(canonical_metric, df.index)
        if row_name is None or row_name not in df.index:
            return False

    return True


def discover_available_metrics(
    lang2folds: Mapping[str, Sequence[pd.DataFrame]],
    candidate_metrics: Sequence[str] = tuple(METRIC_ALIASES.keys()),
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Find canonical metrics available in every fold for each language.

    A metric is considered available for a language if:
      - the language has at least one fold file
      - at least two methods are common across folds
      - the metric is present in every fold, using aliases

    Returns
    -------
    per_lang_available, union_available
    """
    per_lang_available: Dict[str, List[str]] = {}
    union_available = set()

    for lang, folds in lang2folds.items():
        folds = list(folds)
        if not folds:
            per_lang_available[lang] = []
            continue

        common_methods = common_methods_across_folds(folds)
        if len(common_methods) < 2:
            per_lang_available[lang] = []
            continue

        available = [metric for metric in candidate_metrics if metric_present_in_all_folds(folds, metric)]
        per_lang_available[lang] = sorted(available)
        union_available.update(available)

    return per_lang_available, sorted(union_available)


# ---------------------------------------------------------------------
# Vector construction
# ---------------------------------------------------------------------

def concat_metric_vector_across_folds_and_methods(
    fold_dfs: Sequence[pd.DataFrame],
    canonical_metric: str,
) -> Optional[pd.Series]:
    """Concatenate one metric's method values across folds.

    The returned Series has a MultiIndex:

        index = (fold, method)

    It uses only methods present in every fold, preserving the first fold's method
    order.
    """
    if not fold_dfs:
        return None

    common_methods = common_methods_across_folds(fold_dfs)
    if len(common_methods) < 2:
        return None

    vectors: List[pd.Series] = []
    for fold_id, df in enumerate(fold_dfs):
        row_name = first_present(canonical_metric, df.index)
        if row_name is None or row_name not in df.index:
            return None

        series = df.loc[row_name, common_methods].copy()
        series.index = pd.MultiIndex.from_product([[fold_id], common_methods], names=["fold", "method"])
        vectors.append(series)

    return pd.concat(vectors, axis=0)


def aligned_metric_vectors(
    fold_dfs: Sequence[pd.DataFrame],
    metric_1: str,
    metric_2: str,
) -> Optional[pd.DataFrame]:
    """Return aligned finite vectors for two metrics.

    Returns a DataFrame with columns ['x', 'y'] and index (fold, method), or None
    if either metric is unavailable.
    """
    v1 = concat_metric_vector_across_folds_and_methods(fold_dfs, metric_1)
    v2 = concat_metric_vector_across_folds_and_methods(fold_dfs, metric_2)

    if v1 is None or v2 is None:
        return None

    aligned = pd.concat([v1, v2], axis=1)
    aligned.columns = ["x", "y"]
    aligned = aligned.dropna()
    return aligned


# ---------------------------------------------------------------------
# Pairwise correlations
# ---------------------------------------------------------------------

def compute_pair_correlation(
    x: Iterable[float],
    y: Iterable[float],
) -> Dict[str, float]:
    """Compute Kendall tau, Kendall p-value, Pearson r, and n for two vectors."""
    x_arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float).ravel()
    y_arr = np.asarray(list(y) if not isinstance(y, np.ndarray) else y, dtype=float).ravel()

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length.")

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = int(x_arr.size)

    result = {
        "kendall_tau": np.nan,
        "kendall_p": np.nan,
        "pearson_r": np.nan,
        "n": n,
    }

    if n < 2:
        return result

    try:
        tau, tau_p = kendalltau(x_arr, y_arr, nan_policy="omit")
        result["kendall_tau"] = float(tau)
        result["kendall_p"] = float(tau_p) if np.isfinite(tau_p) else np.nan
    except Exception:
        pass

    try:
        r, _ = pearsonr(x_arr, y_arr)
        result["pearson_r"] = float(r)
    except Exception:
        pass

    return result


def compute_corr_for_pair_per_language(
    lang2folds: Mapping[str, Sequence[pd.DataFrame]],
    metric_1: str,
    metric_2: str,
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise correlations for one metric pair in each language.

    Returns
    -------
    dict
        lang -> {'kendall_tau', 'kendall_p', 'pearson_r', 'n'}
    """
    results: Dict[str, Dict[str, float]] = {}

    for lang, folds in lang2folds.items():
        aligned = aligned_metric_vectors(list(folds), metric_1, metric_2)

        if aligned is None or aligned.empty:
            results[lang] = {
                "kendall_tau": np.nan,
                "kendall_p": np.nan,
                "pearson_r": np.nan,
                "n": 0,
            }
            continue

        results[lang] = compute_pair_correlation(aligned["x"], aligned["y"])

    return results


def metric_pairs_from_available_metrics(metrics: Sequence[str]) -> List[Tuple[str, str]]:
    """Return all unique metric pairs from a metric list."""
    return list(combinations(list(metrics), 2))


# ---------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------

def build_correlation_tables(
    lang2folds: Mapping[str, Sequence[pd.DataFrame]],
    use_all_metrics: bool = True,
    metric_pairs_by_group: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
    candidate_metrics: Sequence[str] = tuple(METRIC_ALIASES.keys()),
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, List[str]],
]:
    """Build pairwise correlation tables.

    Returns
    -------
    tau_tables, tau_p_tables, r_tables, n_tables, per_lang_available
    """
    lang2folds = {lang: list(folds) for lang, folds in lang2folds.items()}
    languages = list(lang2folds.keys())

    per_lang_available, union_available = discover_available_metrics(lang2folds, candidate_metrics)

    if metric_pairs_by_group is None:
        if use_all_metrics:
            metric_pairs_by_group = {
                "All Metrics": metric_pairs_from_available_metrics(union_available),
            }
        else:
            metric_pairs_by_group = CURATED_METRIC_PAIRS_BY_GROUP

    tau_tables: Dict[str, pd.DataFrame] = {}
    tau_p_tables: Dict[str, pd.DataFrame] = {}
    r_tables: Dict[str, pd.DataFrame] = {}
    n_tables: Dict[str, pd.DataFrame] = {}

    for group_name, pairs in metric_pairs_by_group.items():
        pairs = list(pairs)
        index = pd.MultiIndex.from_tuples(pairs, names=["metric_1", "metric_2"])

        tau_table = pd.DataFrame(index=index, columns=languages, dtype=float)
        tau_p_table = pd.DataFrame(index=index, columns=languages, dtype=float)
        r_table = pd.DataFrame(index=index, columns=languages, dtype=float)
        n_table = pd.DataFrame(index=index, columns=languages, dtype=float)

        for metric_1, metric_2 in pairs:
            per_lang = compute_corr_for_pair_per_language(lang2folds, metric_1, metric_2)
            for lang in languages:
                values = per_lang.get(lang, {})
                tau_table.loc[(metric_1, metric_2), lang] = values.get("kendall_tau", np.nan)
                tau_p_table.loc[(metric_1, metric_2), lang] = values.get("kendall_p", np.nan)
                r_table.loc[(metric_1, metric_2), lang] = values.get("pearson_r", np.nan)
                n_table.loc[(metric_1, metric_2), lang] = values.get("n", 0)

        tau_tables[group_name] = tau_table
        tau_p_tables[group_name] = tau_p_table
        r_tables[group_name] = r_table
        n_tables[group_name] = n_table

    return tau_tables, tau_p_tables, r_tables, n_tables, per_lang_available


def flatten_pair_tables(
    tau_tables: Mapping[str, pd.DataFrame],
    tau_p_tables: Mapping[str, pd.DataFrame],
    r_tables: Mapping[str, pd.DataFrame],
    n_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Convert grouped pairwise tables into one tidy DataFrame."""
    rows: List[Dict[str, object]] = []

    for group, tau_table in tau_tables.items():
        tau_p_table = tau_p_tables[group]
        r_table = r_tables[group]
        n_table = n_tables[group]

        for metric_1, metric_2 in tau_table.index:
            for lang in tau_table.columns:
                rows.append(
                    {
                        "group": group,
                        "metric_1": metric_1,
                        "metric_2": metric_2,
                        "language": lang,
                        "kendall_tau": tau_table.loc[(metric_1, metric_2), lang],
                        "kendall_p": tau_p_table.loc[(metric_1, metric_2), lang],
                        "pearson_r": r_table.loc[(metric_1, metric_2), lang],
                        "n": n_table.loc[(metric_1, metric_2), lang],
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------

def correlation_matrix_from_pair_table(
    pair_table: pd.DataFrame,
    metrics: Sequence[str],
    language: str,
) -> Optional[pd.DataFrame]:
    """Convert a pairwise correlation table into a square matrix.

    Parameters
    ----------
    pair_table:
        DataFrame indexed by (metric_1, metric_2), with languages as columns.
    metrics:
        Metrics to include in the square matrix.
    language:
        Language column to use.
    """
    metrics = list(metrics)
    if len(metrics) < 2 or language not in pair_table.columns:
        return None

    matrix = pd.DataFrame(index=metrics, columns=metrics, dtype=float)

    for metric in metrics:
        matrix.loc[metric, metric] = 1.0

    for metric_1, metric_2 in combinations(metrics, 2):
        if (metric_1, metric_2) in pair_table.index:
            value = pair_table.loc[(metric_1, metric_2), language]
        elif (metric_2, metric_1) in pair_table.index:
            value = pair_table.loc[(metric_2, metric_1), language]
        else:
            value = np.nan

        matrix.loc[metric_1, metric_2] = value
        matrix.loc[metric_2, metric_1] = value

    return matrix


def build_language_matrices(
    pair_tables: Mapping[str, pd.DataFrame],
    per_lang_available: Mapping[str, Sequence[str]],
    group_name: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Build square correlation matrices for each language.

    Parameters
    ----------
    pair_tables:
        Group -> pairwise correlation table.
    per_lang_available:
        Language -> available metrics.
    group_name:
        Which group to use. If None, uses the first group.
    """
    if not pair_tables:
        return {}

    if group_name is None:
        group_name = next(iter(pair_tables))

    if group_name not in pair_tables:
        raise ValueError(f"Group {group_name!r} not found. Available groups: {list(pair_tables)}")

    pair_table = pair_tables[group_name]
    matrices: Dict[str, pd.DataFrame] = {}

    for lang, metrics in per_lang_available.items():
        matrix = correlation_matrix_from_pair_table(pair_table, list(metrics), lang)
        if matrix is not None:
            matrices[lang] = matrix

    return matrices


# ---------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------

def analyse_metric_correlations(
    lang_globs: Mapping[str, str],
    use_all_metrics: bool = True,
    metric_pairs_by_group: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
) -> Dict[str, object]:
    """Load fold summaries and compute all correlation outputs.

    Returns a dictionary containing:
      - lang2folds
      - tau_tables
      - tau_p_tables
      - r_tables
      - n_tables
      - per_lang_available
      - tidy
      - tau_matrices
      - r_matrices
    """
    lang2folds = load_all_languages(lang_globs)
    tau_tables, tau_p_tables, r_tables, n_tables, per_lang_available = build_correlation_tables(
        lang2folds,
        use_all_metrics=use_all_metrics,
        metric_pairs_by_group=metric_pairs_by_group,
    )

    tidy = flatten_pair_tables(tau_tables, tau_p_tables, r_tables, n_tables)
    tau_matrices = build_language_matrices(tau_tables, per_lang_available)
    r_matrices = build_language_matrices(r_tables, per_lang_available)

    return {
        "lang2folds": lang2folds,
        "tau_tables": tau_tables,
        "tau_p_tables": tau_p_tables,
        "r_tables": r_tables,
        "n_tables": n_tables,
        "per_lang_available": per_lang_available,
        "tidy": tidy,
        "tau_matrices": tau_matrices,
        "r_matrices": r_matrices,
    }


__all__ = [
    "METRIC_ALIASES",
    "CURATED_METRIC_PAIRS_BY_GROUP",
    "first_present",
    "load_language_folds",
    "load_all_languages",
    "common_methods_across_folds",
    "metric_present_in_all_folds",
    "discover_available_metrics",
    "concat_metric_vector_across_folds_and_methods",
    "aligned_metric_vectors",
    "compute_pair_correlation",
    "compute_corr_for_pair_per_language",
    "metric_pairs_from_available_metrics",
    "build_correlation_tables",
    "flatten_pair_tables",
    "correlation_matrix_from_pair_table",
    "build_language_matrices",
    "analyse_metric_correlations",
]
