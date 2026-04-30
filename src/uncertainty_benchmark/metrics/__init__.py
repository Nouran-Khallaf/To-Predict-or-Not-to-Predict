"""Metric utilities for uncertainty benchmarking.

The metrics package is organised by metric family:

- calibration: ECE, CITL, C-Slope
- discrimination: ROC-AUC and AU-PRC error detection
- selective_prediction: risk-coverage and threshold-improvement metrics
- rejection: fixed-rate rejection analysis
- suite: combined metric runner
"""

from uncertainty_benchmark.metrics.calibration import (
    calibration_bins,
    calibration_in_the_large,
    calibration_slope,
    calibration_slope_and_intercept,
    compute_calibration_for_methods,
    compute_calibration_from_correctness,
    compute_calibration_metrics,
    compute_ece,
    confidence_from_uncertainty,
    expected_calibration_error,
    mean_calibration_bias,
    uncertainty_to_confidence,
)
from uncertainty_benchmark.metrics.discrimination import (
    auprc_uncertainty,
    compute_discrimination_for_methods,
    compute_discrimination_from_errors,
    compute_discrimination_metrics,
    roc_auc_uncertainty,
)
from uncertainty_benchmark.metrics.rejection import (
    DEFAULT_REJECTION_RATES,
    DEFAULT_REVERSE_SCORE_METHODS,
    baseline_error_rate_pct,
    baseline_macro_f1,
    compute_rejection_for_methods,
    compute_rejection_metrics,
    compute_rejection_summary_arrays,
    macro_f1_after_rejection,
    macro_f1_delta_after_rejection,
    pct_incorrect_rejected,
    rejected_error_counts,
    rejection_count,
    rejection_results_to_rows,
    split_rejected_kept_indices,
)
from uncertainty_benchmark.metrics.risk_coverage import (
    compute_rc_metrics,
    macro_f1_risk_curve,
)
from uncertainty_benchmark.metrics.selective_prediction import (
    compute_selective_prediction_for_methods,
    compute_selective_prediction_from_errors,
    compute_selective_prediction_metrics,
    e_auopt_rc_from_errors,
    normalised_rc_auc_from_errors,
    optimal_rc_auc_from_errors,
    random_rc_auc_from_errors,
    rc_auc_from_errors,
    risk_at_coverage,
    risk_at_coverage_from_errors,
    risk_coverage_curve,
    risk_coverage_curve_from_errors,
    threshold_improvement,
    threshold_improvement_from_errors,
)
from uncertainty_benchmark.metrics.suite import (
    METRIC_NAMES,
    compute_metrics_per_method_with_timing,
    compute_single_method_metrics,
    metrics_to_long,
    safe_calibration_slope,
    safe_citl,
)

__all__ = [
    # Calibration
    "calibration_bins",
    "calibration_in_the_large",
    "calibration_slope",
    "calibration_slope_and_intercept",
    "compute_calibration_for_methods",
    "compute_calibration_from_correctness",
    "compute_calibration_metrics",
    "compute_ece",
    "confidence_from_uncertainty",
    "expected_calibration_error",
    "mean_calibration_bias",
    "uncertainty_to_confidence",
    # Discrimination
    "auprc_uncertainty",
    "compute_discrimination_for_methods",
    "compute_discrimination_from_errors",
    "compute_discrimination_metrics",
    "roc_auc_uncertainty",
    # Rejection
    "DEFAULT_REJECTION_RATES",
    "DEFAULT_REVERSE_SCORE_METHODS",
    "baseline_error_rate_pct",
    "baseline_macro_f1",
    "compute_rejection_for_methods",
    "compute_rejection_metrics",
    "compute_rejection_summary_arrays",
    "macro_f1_after_rejection",
    "macro_f1_delta_after_rejection",
    "pct_incorrect_rejected",
    "rejected_error_counts",
    "rejection_count",
    "rejection_results_to_rows",
    "split_rejected_kept_indices",
    # Risk coverage compatibility
    "compute_rc_metrics",
    "macro_f1_risk_curve",
    # Selective prediction
    "compute_selective_prediction_for_methods",
    "compute_selective_prediction_from_errors",
    "compute_selective_prediction_metrics",
    "e_auopt_rc_from_errors",
    "normalised_rc_auc_from_errors",
    "optimal_rc_auc_from_errors",
    "random_rc_auc_from_errors",
    "rc_auc_from_errors",
    "risk_at_coverage",
    "risk_at_coverage_from_errors",
    "risk_coverage_curve",
    "risk_coverage_curve_from_errors",
    "threshold_improvement",
    "threshold_improvement_from_errors",
    # Suite
    "METRIC_NAMES",
    "compute_metrics_per_method_with_timing",
    "compute_single_method_metrics",
    "metrics_to_long",
    "safe_calibration_slope",
    "safe_citl",
]