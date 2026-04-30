#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uncertainty_benchmark.analysis.statistical_tests

Reusable statistical-test helpers for fold-level uncertainty benchmark analysis.

This module centralises the logic used by reporting scripts to compare methods
across folds.

Main features
-------------
- Paired best-vs-other tests
- Shapiro-Wilk normality check on paired differences
- Paired t-test when differences are approximately normal
- Wilcoxon signed-rank test otherwise
- Safe handling of too few folds, NaNs, and identical vectors
- Identification of methods statistically tied with the best
- Optional practical-tolerance filtering for ties

Typical usage
-------------
>>> from uncertainty_benchmark.analysis.statistical_tests import compare_best_vs_others
>>> pairwise_df, close_methods = compare_best_vs_others(stacked, best_method, alpha=0.05)

Expected stacked format
-----------------------
Most functions expect a DataFrame like this:

    index = MultiIndex(fold, metric)
    columns = method names
    values = metric values

For example:

    method              SR       SMP       ENT
    fold metric
    0    ROC-AUC       0.71     0.69      0.73
    1    ROC-AUC       0.72     0.68      0.74
    0    ECE           0.12     0.15      0.10
    1    ECE           0.11     0.14      0.09
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PairedTestResult:
    """Container for a paired statistical test result."""

    test: str
    statistic: float
    p_value: float
    normality_p: float
    n: int
    mean_difference: float


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def paired_finite_values(
    x: Iterable[float],
    y: Iterable[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite paired values from two arrays."""
    x_arr = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float).ravel()
    y_arr = np.asarray(list(y) if not isinstance(y, np.ndarray) else y, dtype=float).ravel()

    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Paired vectors must have the same length. Got {x_arr.shape[0]} and {y_arr.shape[0]}.")

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def safe_shapiro(differences: Iterable[float]) -> float:
    """Run Shapiro-Wilk normality test safely.

    Returns NaN when the test is not applicable.
    """
    diffs = np.asarray(list(differences) if not isinstance(differences, np.ndarray) else differences, dtype=float).ravel()
    diffs = diffs[np.isfinite(diffs)]

    if diffs.size < 3:
        return float("nan")

    if np.allclose(diffs, diffs[0], atol=0.0, rtol=0.0):
        return float("nan")

    try:
        return float(shapiro(diffs).pvalue)
    except Exception:
        return float("nan")


def safe_wilcoxon_pair(
    x: Iterable[float],
    y: Iterable[float],
    alternative: str = "two-sided",
) -> PairedTestResult:
    """Run paired Wilcoxon signed-rank test safely."""
    x_arr, y_arr = paired_finite_values(x, y)
    n = int(x_arr.size)

    if n < 2:
        return PairedTestResult("N/A", np.nan, np.nan, np.nan, n, np.nan)

    diffs = x_arr - y_arr
    mean_diff = float(np.mean(diffs))

    if np.allclose(diffs, 0.0, atol=0.0, rtol=0.0):
        return PairedTestResult("No difference", 0.0, 1.0, np.nan, n, mean_diff)

    try:
        stat, p_value = wilcoxon(
            diffs,
            zero_method="zsplit",
            alternative=alternative,
            correction=False,
            mode="auto",
        )
        return PairedTestResult("Wilcoxon", float(stat), float(p_value), np.nan, n, mean_diff)
    except Exception:
        return PairedTestResult("Wilcoxon", np.nan, np.nan, np.nan, n, mean_diff)


def safe_paired_ttest(
    x: Iterable[float],
    y: Iterable[float],
    alternative: str = "two-sided",
) -> PairedTestResult:
    """Run paired t-test safely."""
    x_arr, y_arr = paired_finite_values(x, y)
    n = int(x_arr.size)

    if n < 2:
        return PairedTestResult("N/A", np.nan, np.nan, np.nan, n, np.nan)

    diffs = x_arr - y_arr
    mean_diff = float(np.mean(diffs))

    if np.allclose(diffs, 0.0, atol=0.0, rtol=0.0):
        return PairedTestResult("No difference", 0.0, 1.0, np.nan, n, mean_diff)

    try:
        stat, p_value = ttest_rel(x_arr, y_arr, nan_policy="omit", alternative=alternative)
        return PairedTestResult("Paired t-test", float(stat), float(p_value), np.nan, n, mean_diff)
    except TypeError:
        # Older scipy versions may not support the alternative argument.
        try:
            stat, p_value = ttest_rel(x_arr, y_arr, nan_policy="omit")
            return PairedTestResult("Paired t-test", float(stat), float(p_value), np.nan, n, mean_diff)
        except Exception:
            return PairedTestResult("Paired t-test", np.nan, np.nan, np.nan, n, mean_diff)
    except Exception:
        return PairedTestResult("Paired t-test", np.nan, np.nan, np.nan, n, mean_diff)


def paired_test_auto(
    x: Iterable[float],
    y: Iterable[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    normality_min_n: int = 3,
) -> PairedTestResult:
    """Run a paired test with automatic test selection.

    Test-selection rule:
      - Compute paired differences x - y.
      - If n >= normality_min_n and Shapiro p >= alpha, use paired t-test.
      - Otherwise, use Wilcoxon signed-rank test.

    Parameters
    ----------
    x, y:
        Paired metric values, usually across folds.
    alpha:
        Normality threshold and later significance threshold.
    alternative:
        'two-sided', 'greater', or 'less'.
    normality_min_n:
        Minimum number of pairs required for Shapiro-Wilk.

    Returns
    -------
    PairedTestResult
    """
    x_arr, y_arr = paired_finite_values(x, y)
    n = int(x_arr.size)

    if n < 2:
        return PairedTestResult("N/A", np.nan, np.nan, np.nan, n, np.nan)

    diffs = x_arr - y_arr
    mean_diff = float(np.mean(diffs))

    if np.allclose(diffs, 0.0, atol=0.0, rtol=0.0):
        return PairedTestResult("No difference", 0.0, 1.0, np.nan, n, mean_diff)

    normality_p = float("nan")
    if n >= normality_min_n:
        normality_p = safe_shapiro(diffs)

    if np.isfinite(normality_p) and normality_p >= alpha:
        result = safe_paired_ttest(x_arr, y_arr, alternative=alternative)
        return PairedTestResult(
            result.test,
            result.statistic,
            result.p_value,
            normality_p,
            result.n,
            result.mean_difference,
        )

    result = safe_wilcoxon_pair(x_arr, y_arr, alternative=alternative)
    return PairedTestResult(
        result.test,
        result.statistic,
        result.p_value,
        normality_p,
        result.n,
        result.mean_difference,
    )


def safe_wilcoxon_against_baseline(
    values: Iterable[float],
    baseline: float | Iterable[float],
    alternative: str = "two-sided",
) -> PairedTestResult:
    """Run Wilcoxon test against a scalar or paired baseline vector."""
    values_arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float).ravel()

    if np.ndim(baseline) == 0:
        baseline_arr = np.full_like(values_arr, float(baseline), dtype=float)
    else:
        baseline_arr = np.asarray(list(baseline) if not isinstance(baseline, np.ndarray) else baseline, dtype=float).ravel()

    return safe_wilcoxon_pair(values_arr, baseline_arr, alternative=alternative)


# ---------------------------------------------------------------------
# Best method helpers
# ---------------------------------------------------------------------

def choose_best_by_rule(
    values: Mapping[str, float] | pd.Series,
    higher_is_better: bool = False,
    target_value: Optional[float] = None,
) -> str:
    """Choose best method from method -> mean value.

    Parameters
    ----------
    values:
        Mapping or Series of method means.
    higher_is_better:
        If True, choose the maximum value.
    target_value:
        If provided, choose the method closest to this target value. This is
        useful for C-Slope, where target is 1, and CITL, where target is 0.
    """
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return ""

    if target_value is not None:
        return str((s - target_value).abs().idxmin())

    if higher_is_better:
        return str(s.idxmax())

    return str(s.idxmin())


def best_methods_from_summary(
    mean_values: pd.DataFrame,
    higher_is_better_metrics: Iterable[str],
    target_value_metrics: Optional[Mapping[str, float]] = None,
) -> Dict[str, str]:
    """Choose the best method for each metric from a mean-value table."""
    higher_set = set(higher_is_better_metrics)
    target_value_metrics = dict(target_value_metrics or {})

    best: Dict[str, str] = {}
    for metric in mean_values.index:
        best[metric] = choose_best_by_rule(
            mean_values.loc[metric],
            higher_is_better=(metric in higher_set),
            target_value=target_value_metrics.get(metric),
        )
    return best


# ---------------------------------------------------------------------
# Best-vs-other comparisons
# ---------------------------------------------------------------------

def compare_metric_best_vs_others(
    metric_values: pd.DataFrame,
    metric: str,
    best_method: str,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Compare the best method against every other method for one metric.

    Parameters
    ----------
    metric_values:
        DataFrame with rows = folds and columns = methods.
    metric:
        Metric name.
    best_method:
        Name of the best method.
    alpha:
        Significance threshold.

    Returns
    -------
    records, close_methods:
        records is a list of tidy dictionaries. close_methods contains methods
        with p >= alpha, i.e. not significantly different from the best.
    """
    if best_method not in metric_values.columns:
        return [], []

    records: List[Dict[str, object]] = []
    close_methods: List[str] = []

    for other_method in metric_values.columns:
        if other_method == best_method:
            continue

        result = paired_test_auto(
            metric_values[best_method],
            metric_values[other_method],
            alpha=alpha,
            alternative=alternative,
        )

        is_close = bool(np.isfinite(result.p_value) and result.p_value >= alpha)
        if is_close:
            close_methods.append(str(other_method))

        records.append(
            {
                "metric": metric,
                "best_method": best_method,
                "other_method": str(other_method),
                "test": result.test,
                "statistic": result.statistic,
                "normality_p": result.normality_p,
                "p_value": result.p_value,
                "n_pairs": result.n,
                "mean_difference_best_minus_other": result.mean_difference,
                "not_significantly_different_from_best": is_close,
            }
        )

    return records, close_methods


def compare_best_vs_others(
    stacked: pd.DataFrame,
    best_method: Mapping[str, str],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    metric_level: str = "metric",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Compare best method with all other methods for every metric.

    Parameters
    ----------
    stacked:
        MultiIndex DataFrame with one index level named `metric` and columns as
        methods. Usually index = (fold, metric).
    best_method:
        Mapping from metric name to best method.
    alpha:
        Significance threshold.
    alternative:
        Alternative hypothesis for paired tests.
    metric_level:
        Name of the metric level in the stacked DataFrame.

    Returns
    -------
    pairwise_df, close_methods
    """
    if not isinstance(stacked.index, pd.MultiIndex):
        raise ValueError("stacked must have a MultiIndex, usually (fold, metric).")

    if metric_level not in stacked.index.names:
        raise ValueError(f"Metric level {metric_level!r} not found in index names {stacked.index.names}.")

    all_records: List[Dict[str, object]] = []
    close_methods: Dict[str, List[str]] = {}

    metrics = list(stacked.index.get_level_values(metric_level).unique())

    for metric in metrics:
        best = best_method.get(metric, "")
        if not best:
            close_methods[metric] = []
            continue

        metric_df = stacked.xs(metric, level=metric_level).astype(float)
        records, close = compare_metric_best_vs_others(
            metric_values=metric_df,
            metric=str(metric),
            best_method=str(best),
            alpha=alpha,
            alternative=alternative,
        )
        all_records.extend(records)
        close_methods[str(metric)] = close

    return pd.DataFrame.from_records(all_records), close_methods


# ---------------------------------------------------------------------
# Ties with best for array/tensor summaries
# ---------------------------------------------------------------------

def pairwise_tie_mask_vs_best(
    data_tensor: np.ndarray,
    mean_matrix: np.ndarray,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find best indices and methods statistically tied with best.

    Parameters
    ----------
    data_tensor:
        Array with shape methods x folds x thresholds.
    mean_matrix:
        Array with shape methods x thresholds.
    alpha:
        Significance threshold. A method is tied with best when p >= alpha.
    higher_is_better:
        If True, best is max. If False, best is min.

    Returns
    -------
    best_idx, tied_mask
        best_idx has shape thresholds.
        tied_mask has shape methods x thresholds.
    """
    data_tensor = np.asarray(data_tensor, dtype=float)
    mean_matrix = np.asarray(mean_matrix, dtype=float)

    if data_tensor.ndim != 3:
        raise ValueError("data_tensor must have shape methods x folds x thresholds.")
    if mean_matrix.ndim != 2:
        raise ValueError("mean_matrix must have shape methods x thresholds.")

    n_methods, _, n_thresholds = data_tensor.shape
    if mean_matrix.shape != (n_methods, n_thresholds):
        raise ValueError("mean_matrix shape must match data_tensor methods and thresholds.")

    best_idx = np.full(n_thresholds, -1, dtype=int)
    tied_mask = np.zeros((n_methods, n_thresholds), dtype=bool)

    for threshold_idx in range(n_thresholds):
        means = mean_matrix[:, threshold_idx]
        if not np.any(np.isfinite(means)):
            continue

        if higher_is_better:
            best = int(np.nanargmax(means))
        else:
            best = int(np.nanargmin(means))

        best_idx[threshold_idx] = best
        best_values = data_tensor[best, :, threshold_idx]

        for method_idx in range(n_methods):
            if method_idx == best:
                continue

            other_values = data_tensor[method_idx, :, threshold_idx]
            result = safe_wilcoxon_pair(other_values, best_values)
            if np.isfinite(result.p_value) and result.p_value >= alpha:
                tied_mask[method_idx, threshold_idx] = True

    return best_idx, tied_mask


def refine_ties_with_tolerance(
    mean_matrix: np.ndarray,
    best_idx: np.ndarray,
    tied_mask: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Keep only statistical ties that are practically close to the best mean.

    Parameters
    ----------
    mean_matrix:
        Array with shape methods x thresholds.
    best_idx:
        Best method index for each threshold.
    tied_mask:
        Boolean tie mask from pairwise_tie_mask_vs_best.
    tolerance:
        Maximum absolute difference from the best mean.
    """
    mean_matrix = np.asarray(mean_matrix, dtype=float)
    best_idx = np.asarray(best_idx, dtype=int)
    tied_mask = np.asarray(tied_mask, dtype=bool)

    refined = np.zeros_like(tied_mask, dtype=bool)
    n_methods, n_thresholds = mean_matrix.shape

    if tied_mask.shape != (n_methods, n_thresholds):
        raise ValueError("tied_mask shape must match mean_matrix.")
    if best_idx.shape[0] != n_thresholds:
        raise ValueError("best_idx length must match number of thresholds.")

    for threshold_idx in range(n_thresholds):
        best = best_idx[threshold_idx]
        if best < 0 or not np.isfinite(mean_matrix[best, threshold_idx]):
            continue

        best_mean = mean_matrix[best, threshold_idx]
        diffs = np.abs(mean_matrix[:, threshold_idx] - best_mean)
        keep = tied_mask[:, threshold_idx] & np.isfinite(diffs) & (diffs <= tolerance)
        refined[:, threshold_idx] = keep

    return refined


# ---------------------------------------------------------------------
# Multiple-comparison helper
# ---------------------------------------------------------------------

def bonferroni_correct(p_values: Iterable[float]) -> np.ndarray:
    """Apply Bonferroni correction to p-values."""
    p = np.asarray(list(p_values) if not isinstance(p_values, np.ndarray) else p_values, dtype=float)
    corrected = p * np.sum(np.isfinite(p))
    corrected = np.where(np.isfinite(corrected), np.minimum(corrected, 1.0), np.nan)
    return corrected


def add_bonferroni_column(
    df: pd.DataFrame,
    p_col: str = "p_value",
    group_col: Optional[str] = "metric",
    out_col: str = "p_value_bonferroni",
) -> pd.DataFrame:
    """Add a Bonferroni-corrected p-value column to a pairwise-results DataFrame."""
    result = df.copy()

    if p_col not in result.columns:
        raise ValueError(f"Column {p_col!r} not found.")

    if group_col is None:
        result[out_col] = bonferroni_correct(result[p_col].to_numpy(dtype=float))
        return result

    if group_col not in result.columns:
        raise ValueError(f"Column {group_col!r} not found.")

    result[out_col] = np.nan
    for _, idx in result.groupby(group_col).groups.items():
        corrected = bonferroni_correct(result.loc[idx, p_col].to_numpy(dtype=float))
        result.loc[idx, out_col] = corrected

    return result


__all__ = [
    "PairedTestResult",
    "paired_finite_values",
    "safe_shapiro",
    "safe_wilcoxon_pair",
    "safe_paired_ttest",
    "paired_test_auto",
    "safe_wilcoxon_against_baseline",
    "choose_best_by_rule",
    "best_methods_from_summary",
    "compare_metric_best_vs_others",
    "compare_best_vs_others",
    "pairwise_tie_mask_vs_best",
    "refine_ties_with_tolerance",
    "bonferroni_correct",
    "add_bonferroni_column",
]
