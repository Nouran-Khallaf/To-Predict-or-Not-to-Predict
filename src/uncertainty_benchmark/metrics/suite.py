"""Original-notebook-compatible metric suite.

This file intentionally follows the metric definitions used in the user's
original notebook/code block, not the newer package calibration/selective
prediction definitions.

Score convention
----------------
Input method columns are expected to be the raw scores saved by the original
pipeline:

- SR, SMP, ENT, ENT_MC, PV, BALD, MD, HUQ-MD: larger = more uncertain.
- LOF, ISOF: raw sklearn ``decision_function`` scores are saved; larger means
  more in-distribution / more confident. These are flipped exactly once here
  before normalisation, matching the notebook ``preprocess_data`` step.

Metrics are computed after min-max normalising each uncertainty column to
``u_norm`` and defining ``confidence = 1 - u_norm``.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score


METHOD_ORDER = [
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
    "MARGIN",
]

METRIC_NAMES = [
    "ROC-AUC",
    "AU-PRC",
    "C-Slope",
    "CITL",
    "ECE",
    "RC-AUC",
    "Norm RC-AUC",
    "E-AUoptRC",
    "TI",
    "TI@95",
    "Optimal Coverage",
]

RAW_CONFIDENCE_METHODS = {"LOF", "ISOF"}


def _timer() -> float:
    return time.perf_counter()


def _to_numpy_1d(values, dtype=float) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    return arr.reshape(-1)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        if np.issubdtype(arr.dtype, np.number):
            mask &= np.isfinite(arr)
    return mask


def _normalise_uncertainty(raw: np.ndarray) -> np.ndarray:
    """Min-max normalisation used by the notebook preprocess step."""
    raw = _to_numpy_1d(raw, dtype=float)
    finite = np.isfinite(raw)
    out = np.full_like(raw, np.nan, dtype=float)

    if not finite.any():
        return out

    vals = raw[finite]
    col_min = float(vals.min())
    col_max = float(vals.max())

    if np.isclose(col_max, col_min):
        out[finite] = 0.0
    else:
        out[finite] = (vals - col_min) / (col_max - col_min)

    return out


def _raw_to_uncertainty(raw, method: str | None = None) -> np.ndarray:
    """Convert raw saved scores to the notebook uncertainty convention.

    The original notebook saved raw sklearn decision_function values for LOF
    and ISOF. Those values are higher for more normal / more confident samples,
    so they are negated once during metric preprocessing.
    """
    raw = _to_numpy_1d(raw, dtype=float)
    if method in RAW_CONFIDENCE_METHODS:
        return -raw
    return raw


# ---------------------------------------------------------------------
# Calibration and discrimination metrics from the notebook
# ---------------------------------------------------------------------


def compute_ece(confidence, y_true_bin, n_bins: int = 15) -> float:
    """ECE between confidence and observed correctness.

    This matches the final notebook implementation: it returns the native
    0--1 value, not a percentage.
    """
    confidences = np.clip(_to_numpy_1d(confidence, dtype=float), 0.0, 1.0)
    outcomes = _to_numpy_1d(y_true_bin, dtype=float)

    mask = _finite_mask(confidences, outcomes)
    confidences = confidences[mask]
    outcomes = outcomes[mask]

    n = len(outcomes)
    if n == 0:
        return float("nan")

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i < n_bins - 1:
            in_bin = (confidences >= lo) & (confidences < hi)
        else:
            in_bin = (confidences >= lo) & (confidences <= hi)

        if in_bin.any():
            bin_acc = outcomes[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            weight = in_bin.sum() / n
            ece += weight * abs(bin_acc - bin_conf)

    return float(ece)


def safe_calibration_slope(confidence, y_true_bin) -> float:
    """Notebook C-Slope: scipy.linregress(confidence, correctness)."""
    confidence = _to_numpy_1d(confidence, dtype=float)
    y_true_bin = _to_numpy_1d(y_true_bin, dtype=float)
    mask = _finite_mask(confidence, y_true_bin)

    if mask.sum() < 2 or np.isclose(confidence[mask].max(), confidence[mask].min()):
        return float("nan")

    slope, _, _, _, _ = linregress(confidence[mask], y_true_bin[mask])
    return float(slope)


def safe_citl(confidence, y_true_bin) -> float:
    """Notebook CITL: mean(confidence) - mean(correctness)."""
    confidence = _to_numpy_1d(confidence, dtype=float)
    y_true_bin = _to_numpy_1d(y_true_bin, dtype=float)
    mask = _finite_mask(confidence, y_true_bin)

    if not mask.any():
        return float("nan")

    return float(confidence[mask].mean() - y_true_bin[mask].mean())


# ---------------------------------------------------------------------
# RC-AUC and normalised RC-AUC from the notebook
# ---------------------------------------------------------------------


def rcc_auc(confidence_scores, error_vector, normalize: bool = True, return_points: bool = False):
    """Area under risk-coverage curve from confidence and error labels.

    This follows the notebook function ``rcc_auc``:
    - ``confidence_scores``: higher = more confident.
    - ``error_vector``: 1 = incorrect, 0 = correct.
    - sort by descending confidence.
    - risk at each coverage = cumulative mean error.
    """
    confidence_scores = _to_numpy_1d(confidence_scores, dtype=float)
    error_vector = _to_numpy_1d(error_vector, dtype=float)

    mask = _finite_mask(confidence_scores, error_vector)
    confidence_scores = confidence_scores[mask]
    error_vector = error_vector[mask]

    n = len(error_vector)
    if n == 0:
        if return_points:
            return float("nan"), np.array([]), np.array([])
        return float("nan")

    if normalize:
        c_min = float(confidence_scores.min())
        c_max = float(confidence_scores.max())
        if np.isclose(c_max, c_min):
            confidence_scores = np.zeros_like(confidence_scores, dtype=float)
        else:
            confidence_scores = (confidence_scores - c_min) / (c_max - c_min)

    sorted_indices = np.argsort(-confidence_scores, kind="mergesort")
    sorted_errors = error_vector[sorted_indices]

    cumulative_errors = np.cumsum(sorted_errors)
    coverage = np.arange(1, n + 1, dtype=float) / float(n)
    avg_risk = cumulative_errors / np.arange(1, n + 1, dtype=float)

    rc_value = float(np.trapz(avg_risk, coverage)) if n > 1 else float(avg_risk[0])

    if return_points:
        return rc_value, coverage, avg_risk
    return rc_value


def get_random_scores(function, metrics, num_iter: int = 10, seed: int = 42):
    """Notebook random baseline helper used for Norm RC-AUC."""
    np.random.seed(seed)
    rand_scores = np.arange(len(metrics))

    values = []
    for _ in range(num_iter):
        np.random.shuffle(rand_scores)
        values.append(function(rand_scores, metrics))
    return float(np.mean(values))


def normalized_metric(conf, risk, metric):
    """Notebook normalised metric formula.

    For RC-AUC, this gives values where larger is better and 1 is oracle-like:
        (metric(conf, risk) - random) / (oracle - random)
    with oracle confidence = -risk.
    """
    conf = _to_numpy_1d(conf, dtype=float)
    risk = _to_numpy_1d(risk, dtype=float)
    mask = _finite_mask(conf, risk)
    conf = conf[mask]
    risk = risk[mask]

    if len(risk) == 0:
        return float("nan")

    random_rcauc = get_random_scores(metric, risk)
    oracle_rcauc = metric(-risk, risk)
    denom = oracle_rcauc - random_rcauc

    if np.isclose(denom, 0.0) or not np.isfinite(denom):
        return float("nan")

    return float((metric(conf, risk) - random_rcauc) / denom)


# ---------------------------------------------------------------------
# Macro-F1 selective-prediction metrics from the final notebook block
# ---------------------------------------------------------------------


def compute_rc_metrics(scores_uncert, y_true, y_pred, ti_fixed_cov: float = 0.95):
    """Final notebook RC/TI metrics based on macro-F1 risk.

    Inputs
    ------
    scores_uncert:
        Uncertainty scores; lower = more confident after preprocessing.
    y_true, y_pred:
        Integer class labels.

    Returns
    -------
    e_auopt_rc, ti_cstar, ti_95, coverages, risks, c_star
    """
    scores_uncert = _to_numpy_1d(scores_uncert, dtype=float)
    y_true = _to_numpy_1d(y_true, dtype=int)
    y_pred = _to_numpy_1d(y_pred, dtype=int)

    mask = _finite_mask(scores_uncert, y_true, y_pred)
    scores_uncert = scores_uncert[mask]
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = len(y_true)
    if n == 0:
        empty = np.array([])
        return float("nan"), float("nan"), float("nan"), empty, empty, float("nan")

    order = np.argsort(scores_uncert, kind="mergesort")
    coverages = np.arange(1, n + 1, dtype=float) / float(n)

    risks = []
    for k in range(1, n + 1):
        covered = order[:k]
        f1 = f1_score(y_true[covered], y_pred[covered], average="macro", zero_division=0)
        risks.append(1.0 - f1)
    risks = np.asarray(risks, dtype=float)

    c_star = f1_score(y_true, y_pred, average="macro", zero_division=0)

    k_star = max(1, min(n, int(np.floor(c_star * n))))
    ti_cstar = 1.0 - risks[k_star - 1]
    e_auopt_rc = float(auc(coverages[:k_star], risks[:k_star])) if k_star > 1 else 0.0

    k95 = max(1, min(n, int(np.floor(ti_fixed_cov * n))))
    ti_95 = 1.0 - risks[k95 - 1]

    return float(e_auopt_rc), float(ti_cstar), float(ti_95), coverages, risks, float(c_star)


# Backward-compatible name expected by metrics/__init__.py
compute_rc_metrics_original = compute_rc_metrics


# ---------------------------------------------------------------------
# Combined metric runner
# ---------------------------------------------------------------------


def compute_single_method_metrics(
    uncertainty,
    y_true_bin,
    y_true_idx,
    y_pred_idx,
    method: str | None = None,
    bins: int = 15,
    ti_fixed_coverage: float = 0.95,
) -> Dict[str, float]:
    """Compute all original-notebook metrics for one uncertainty method."""
    raw_uncert = _raw_to_uncertainty(uncertainty, method=method)
    u_norm = _normalise_uncertainty(raw_uncert)
    y_scores = 1.0 - u_norm

    y_true_bin = _to_numpy_1d(y_true_bin, dtype=int)  # 1 correct, 0 incorrect
    y_true_idx = _to_numpy_1d(y_true_idx, dtype=int)
    y_pred_idx = _to_numpy_1d(y_pred_idx, dtype=int)

    mask = _finite_mask(u_norm, y_scores, y_true_bin, y_true_idx, y_pred_idx)
    u_norm = u_norm[mask]
    y_scores = y_scores[mask]
    y_true_bin = y_true_bin[mask]
    y_true_idx = y_true_idx[mask]
    y_pred_idx = y_pred_idx[mask]

    if len(y_true_bin) == 0:
        return {metric: float("nan") for metric in METRIC_NAMES}

    # Calibration
    ece = compute_ece(y_scores, y_true_bin, n_bins=bins)
    c_slope = safe_calibration_slope(y_scores, y_true_bin.astype(float))
    citl = safe_citl(y_scores, y_true_bin.astype(float))

    # Error detection: higher uncertainty => more likely error.
    error_labels = (y_true_bin == 0).astype(int)
    try:
        precision, recall, _ = precision_recall_curve(error_labels, u_norm)
        auprc_e = float(auc(recall, precision))
    except Exception:
        auprc_e = float("nan")

    # Correctness discrimination: higher confidence => more likely correct.
    try:
        rocauc = float(roc_auc_score(y_true_bin, y_scores))
    except Exception:
        rocauc = float("nan")

    # Notebook RC-AUC from confidence and error labels.
    # Notebook RC-AUC from confidence and error labels.

# retained-performance area, so RC-AUC = 1 - AURC.
    try:
        aurc = rcc_auc(y_scores, error_labels)
        rc_auc = 1.0 - aurc
    except Exception:
        aurc = float("nan")
        rc_auc = float("nan")

    try:
        # Keep normalisation based on the original AURC/risk curve.
        norm_rc_auc = normalized_metric(y_scores, error_labels, rcc_auc)
    except Exception:
        norm_rc_auc = float("nan")

    try:
        norm_rc_auc = normalized_metric(y_scores, error_labels, rcc_auc)
    except Exception:
        norm_rc_auc = float("nan")

    # Macro-F1 selective-prediction metrics.
    try:
        e_auopt_rc, ti_cstar, ti_95, _coverages, _risks, c_star = compute_rc_metrics(
            u_norm,
            y_true_idx,
            y_pred_idx,
            ti_fixed_cov=ti_fixed_coverage,
        )
    except Exception:
        e_auopt_rc = ti_cstar = ti_95 = c_star = float("nan")

    return {
        "ROC-AUC": rocauc,
        "AU-PRC": auprc_e,
        "C-Slope": c_slope,
        "CITL": citl,
        "ECE": float(ece),
        "RC-AUC": float(rc_auc),
        "Norm RC-AUC": float(norm_rc_auc),
        "E-AUoptRC": float(e_auopt_rc),
        "TI": float(ti_cstar),
        "TI@95": float(ti_95),
        "Optimal Coverage": float(c_star),
    }


def compute_metrics_per_method_with_timing(
    df: pd.DataFrame,
    methods: list[str],
    bins: int = 15,
    ti_fixed_coverage: float = 0.95,
) -> Tuple[pd.DataFrame, dict[str, float]]:
    """Compute metrics for all methods and record per-method metric time."""
    required = {"correct", "y_true_idx", "y_pred_idx"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for metrics: {sorted(missing)}")

    for method in methods:
        if method not in df.columns:
            raise KeyError(f"Method column not found in dataframe: {method}")

    ordered_methods = [m for m in METHOD_ORDER if m in methods] + [m for m in methods if m not in METHOD_ORDER]

    y_true_bin = df["correct"].astype(int).to_numpy()
    y_true_idx = df["y_true_idx"].astype(int).to_numpy()
    y_pred_idx = df["y_pred_idx"].astype(int).to_numpy()

    results = {metric: {} for metric in METRIC_NAMES}
    method_times: dict[str, float] = {}

    for method in ordered_methods:
        t0 = _timer()
        method_result = compute_single_method_metrics(
            uncertainty=df[method].astype(float).to_numpy(),
            y_true_bin=y_true_bin,
            y_true_idx=y_true_idx,
            y_pred_idx=y_pred_idx,
            method=method,
            bins=bins,
            ti_fixed_coverage=ti_fixed_coverage,
        )
        for metric in METRIC_NAMES:
            results[metric][method] = method_result[metric]
        method_times[method] = _timer() - t0

    metrics_df = pd.DataFrame(results).T
    metrics_df = metrics_df.loc[[m for m in METRIC_NAMES if m in metrics_df.index], ordered_methods]
    return metrics_df, method_times


def metrics_to_long(metrics_df: pd.DataFrame, fold: int) -> pd.DataFrame:
    long_df = metrics_df.copy()
    long_df["fold"] = fold
    long_df["metric"] = long_df.index
    long_df = long_df.reset_index(drop=True)
    return long_df.melt(
        id_vars=["fold", "metric"],
        var_name="method",
        value_name="value",
    )


__all__ = [
    "METHOD_ORDER",
    "METRIC_NAMES",
    "RAW_CONFIDENCE_METHODS",
    "compute_ece",
    "safe_calibration_slope",
    "safe_citl",
    "rcc_auc",
    "get_random_scores",
    "normalized_metric",
    "compute_rc_metrics",
    "compute_rc_metrics_original",
    "compute_single_method_metrics",
    "compute_metrics_per_method_with_timing",
    "metrics_to_long",
]
