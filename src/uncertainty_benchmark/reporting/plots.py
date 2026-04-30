#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uncertainty_benchmark.reporting.plots

Combined plotting helpers for uncertainty benchmark reporting.

This file intentionally replaces both the newer reporting/plots.py and the older
plot/plot2 plotting utilities. Keep only this file inside
src/uncertainty_benchmark/reporting/ to avoid duplicate imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Display defaults
# ---------------------------------------------------------------------

DEFAULT_METHOD_ORDER = [
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
]

DEFAULT_METRIC_ORDER = [
    "ECE",
    "C-Slope",
    "CITL",
    "AU-PRC",
    "ROC-AUC",
    "RC-AUC",
    "Norm RC-AUC",
    "E-AUoptRC",
    "TI",
    "TI@95",
    "Optimal Coverage",
]

METHOD_DISPLAY_NAMES = {
    "ENT_MC": "ENT-MC",
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure parent directory exists and return Path object."""
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    return outpath


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalise_formats(formats: Optional[str | Sequence[str]] = None) -> list[str]:
    """Normalise requested figure formats."""
    if formats is None:
        return ["png"]
    if isinstance(formats, str):
        formats = [formats]

    out: list[str] = []
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".").strip()
        if fmt:
            out.append(fmt)
    return out or ["png"]


def maybe_savefig(
    fig,
    outpath: Optional[str | Path] = None,
    dpi: int = 300,
    bbox_inches: str = "tight",
    close: bool = True,
) -> None:
    """Save figure if outpath is provided."""
    if outpath is None:
        return

    outpath = ensure_parent_dir(outpath)
    fig.savefig(outpath, dpi=dpi, bbox_inches=bbox_inches)
    if close:
        plt.close(fig)


def save_current_figure(path: str | Path, dpi: int = 300, formats: Optional[str | Sequence[str]] = None) -> dict[str, Path]:
    """Save the current matplotlib figure in one or more formats.

    The suffix of ``path`` is replaced by each requested format.
    """
    path = ensure_parent_dir(path)
    formats = normalise_formats(formats)

    plt.tight_layout()
    saved: dict[str, Path] = {}
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        if fmt == "pdf":
            plt.savefig(out_path, bbox_inches="tight")
        else:
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved[fmt] = out_path

    plt.close()
    return saved


def format_method_name(method: str) -> str:
    """Readable method names for figures."""
    return METHOD_DISPLAY_NAMES.get(str(method), str(method))


def _safe_name(text: str) -> str:
    """Make a string safe for filenames."""
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("@", "at")
        .replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def _normalise_long_columns(
    df: pd.DataFrame,
    method_col: str = "method",
    metric_col: str = "metric",
) -> pd.DataFrame:
    """Return a copy with standard ``method`` and ``metric`` columns when possible.

    This keeps old plotting helpers compatible with evaluator outputs that use
    ``mode`` instead of ``method``.
    """
    out = df.copy()
    if method_col not in out.columns:
        if "mode" in out.columns:
            out[method_col] = out["mode"]
        elif "Method" in out.columns:
            out[method_col] = out["Method"]
        elif "Mode" in out.columns:
            out[method_col] = out["Mode"]
    if metric_col not in out.columns:
        if "measure" in out.columns:
            out[metric_col] = out["measure"]
        elif "Metric" in out.columns:
            out[metric_col] = out["Metric"]
    return out


def order_methods(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    """Order rows by the default uncertainty-method order."""
    out = _normalise_long_columns(df, method_col=method_col)
    if method_col not in out.columns:
        raise KeyError(f"Missing method column: {method_col}")

    order = {method: i for i, method in enumerate(DEFAULT_METHOD_ORDER)}
    out["_method_order"] = out[method_col].map(order).fillna(999)
    out = out.sort_values(["_method_order", method_col]).drop(columns=["_method_order"])
    return out


def order_metrics(df: pd.DataFrame, metric_col: str = "metric") -> pd.DataFrame:
    """Order rows by the default metric order."""
    out = _normalise_long_columns(df, metric_col=metric_col)
    if metric_col not in out.columns:
        raise KeyError(f"Missing metric column: {metric_col}")

    order = {metric: i for i, metric in enumerate(DEFAULT_METRIC_ORDER)}
    out["_metric_order"] = out[metric_col].map(order).fillna(999)
    out = out.sort_values(["_metric_order", metric_col]).drop(columns=["_metric_order"])
    return out


# ---------------------------------------------------------------------
# Correlation heatmaps
# ---------------------------------------------------------------------

def plot_correlation_heatmap(
    matrix: pd.DataFrame,
    title: Optional[str] = None,
    outpath: Optional[str | Path] = None,
    dpi: int = 300,
    annotate: bool = True,
    vmin: float = -1.0,
    vmax: float = 1.0,
    colorbar_label: str = "Correlation",
    close: bool = True,
):
    """Plot a square heatmap for a correlation matrix."""
    if matrix.empty:
        raise ValueError("matrix is empty.")

    metrics = list(matrix.index)
    matrix = matrix.loc[metrics, metrics]

    size = max(6.0, 0.55 * len(metrics))
    fig, ax = plt.subplots(figsize=(size, size))

    im = ax.imshow(matrix.values.astype(float), interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(metrics)

    if annotate and len(metrics) <= 12:
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{float(value):.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="both", length=0)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    maybe_savefig(fig, outpath=outpath, dpi=dpi, close=close)
    return fig, ax


def save_correlation_heatmaps(
    matrices: Mapping[str, pd.DataFrame],
    outdir: str | Path,
    prefix: str = "tau_heatmap",
    title_prefix: str = "Kendall's $\\tau$",
    dpi: int = 300,
    annotate: bool = True,
    extension: str = "png",
) -> Dict[str, Path]:
    """Save one heatmap per language/corpus matrix."""
    outdir = ensure_dir(outdir)
    paths: Dict[str, Path] = {}
    for lang, matrix in matrices.items():
        outpath = outdir / f"{prefix}_{lang}.{extension}"
        plot_correlation_heatmap(
            matrix,
            title=f"{title_prefix} — {lang}",
            outpath=outpath,
            dpi=dpi,
            annotate=annotate,
            close=True,
        )
        paths[lang] = outpath
    return paths


# ---------------------------------------------------------------------
# Risk-coverage plots
# ---------------------------------------------------------------------

def plot_risk_coverage_curve(
    coverages: Sequence[float],
    risks: Sequence[float],
    label: Optional[str] = None,
    ax=None,
    title: Optional[str] = None,
    outpath: Optional[str | Path] = None,
    dpi: int = 300,
    close: bool = True,
):
    """Plot a single risk-coverage curve."""
    coverages = np.asarray(coverages, dtype=float)
    risks = np.asarray(risks, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    order = np.argsort(coverages)
    ax.plot(coverages[order], risks[order], marker="o", markersize=2, linewidth=1.5, label=label)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linewidth=0.5, alpha=0.3)

    if title:
        ax.set_title(title)
    if label:
        ax.legend(frameon=False)

    fig.tight_layout()
    maybe_savefig(fig, outpath=outpath, dpi=dpi, close=close)
    return fig, ax


def plot_multiple_risk_coverage_curves(
    curves: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    title: Optional[str] = None,
    outpath: Optional[str | Path] = None,
    dpi: int = 300,
    close: bool = True,
):
    """Plot multiple risk-coverage curves on one axis."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for method, (coverages, risks) in curves.items():
        coverages = np.asarray(coverages, dtype=float)
        risks = np.asarray(risks, dtype=float)
        order = np.argsort(coverages)
        ax.plot(
            coverages[order],
            risks[order],
            marker="o",
            markersize=2,
            linewidth=1.4,
            label=format_method_name(method),
        )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    maybe_savefig(fig, outpath=outpath, dpi=dpi, close=close)
    return fig, ax


# ---------------------------------------------------------------------
# Rejection-rate plots
# ---------------------------------------------------------------------

def plot_rejection_metric_curve(
    summary_df: pd.DataFrame,
    metric_col: str,
    y_label: str,
    title: Optional[str] = None,
    method_col: str = "method",
    rate_col: str = "rejection_rate",
    outpath: Optional[str | Path] = None,
    dpi: int = 300,
    close: bool = True,
):
    """Plot a metric as a function of rejection rate for all methods."""
    summary_df = _normalise_long_columns(summary_df, method_col=method_col)
    required = {method_col, rate_col, metric_col}
    missing = required.difference(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df is missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    for method, sub in summary_df.groupby(method_col):
        sub = sub.sort_values(rate_col)
        ax.plot(
            sub[rate_col].to_numpy(dtype=float) * 100.0,
            sub[metric_col].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            label=format_method_name(str(method)),
        )

    ax.set_xlabel("Rejected examples (%)")
    ax.set_ylabel(y_label)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    maybe_savefig(fig, outpath=outpath, dpi=dpi, close=close)
    return fig, ax


def plot_macro_delta_curve(
    summary_df: pd.DataFrame,
    outpath: Optional[str | Path] = None,
    title: Optional[str] = "Macro-F1 change after rejection",
    dpi: int = 300,
    close: bool = True,
):
    """Plot Macro ΔF1 against rejection rate."""
    return plot_rejection_metric_curve(
        summary_df,
        metric_col="macro_delta_mean_pp",
        y_label="Macro $\\Delta$F1 (pp)",
        title=title,
        outpath=outpath,
        dpi=dpi,
        close=close,
    )


def plot_pct_incorrect_rejected_curve(
    summary_df: pd.DataFrame,
    outpath: Optional[str | Path] = None,
    title: Optional[str] = "% Incorrect among rejected examples",
    dpi: int = 300,
    close: bool = True,
):
    """Plot percentage incorrect among rejected examples against rejection rate."""
    return plot_rejection_metric_curve(
        summary_df,
        metric_col="pct_incorrect_rejected_mean",
        y_label="Incorrect among rejected (%)",
        title=title,
        outpath=outpath,
        dpi=dpi,
        close=close,
    )


# ---------------------------------------------------------------------
# Calibration plots
# ---------------------------------------------------------------------

def plot_reliability_diagram(
    bin_stats: np.ndarray | pd.DataFrame,
    title: Optional[str] = None,
    outpath: Optional[str | Path] = None,
    dpi: int = 300,
    close: bool = True,
):
    """Plot a reliability diagram from calibration-bin statistics.

    Expected columns/order: bin_left, bin_right, n, accuracy, confidence, gap.
    """
    if isinstance(bin_stats, pd.DataFrame):
        if {"accuracy", "confidence"}.issubset(bin_stats.columns):
            accuracy = bin_stats["accuracy"].to_numpy(dtype=float)
            confidence = bin_stats["confidence"].to_numpy(dtype=float)
        else:
            arr = bin_stats.to_numpy(dtype=float)
            accuracy = arr[:, 3]
            confidence = arr[:, 4]
    else:
        arr = np.asarray(bin_stats, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("bin_stats must have at least 5 columns.")
        accuracy = arr[:, 3]
        confidence = arr[:, 4]

    mask = np.isfinite(accuracy) & np.isfinite(confidence)
    accuracy = accuracy[mask]
    confidence = confidence[mask]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot(confidence, accuracy, marker="o", linewidth=1.5, label="Observed")

    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    ax.legend(frameon=False)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    maybe_savefig(fig, outpath=outpath, dpi=dpi, close=close)
    return fig, ax


# ---------------------------------------------------------------------
# Metric summary plots
# ---------------------------------------------------------------------

def plot_metric_by_method(
    mean_values: pd.DataFrame,
    metric: str,
    methods: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    outpath: Optional[str | Path] = None,
    dpi: int = 300,
    close: bool = True,
):
    """Plot a bar chart of one metric across methods from a wide metric table."""
    if metric not in mean_values.index:
        raise ValueError(f"Metric {metric!r} not found in mean_values.")

    if methods is None:
        methods = list(mean_values.columns)
    methods = [m for m in methods if m in mean_values.columns]

    values = mean_values.loc[metric, methods].to_numpy(dtype=float)
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(max(6.0, 0.55 * len(methods)), 4.4))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels([format_method_name(m) for m in methods], rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(metric)

    fig.tight_layout()
    maybe_savefig(fig, outpath=outpath, dpi=dpi, close=close)
    return fig, ax


def plot_metric_summary(
    metrics_summary: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    value_col: str = "mean",
    error_col: str = "std",
    title: Optional[str] = None,
    dpi: int = 300,
    formats: Optional[str | Sequence[str]] = None,
    method_col: str = "method",
    metric_col: str = "metric",
) -> dict[str, Path]:
    """Plot mean metric value by uncertainty method from a long summary CSV."""
    metrics_summary = _normalise_long_columns(metrics_summary, method_col=method_col, metric_col=metric_col)
    required = {method_col, metric_col, value_col}
    missing = required.difference(metrics_summary.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = metrics_summary[metrics_summary[metric_col] == metric].copy()
    df = order_methods(df, method_col=method_col)

    if df.empty:
        raise ValueError(f"No rows found for metric: {metric}")

    x = np.arange(len(df))
    y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)

    plt.figure(figsize=(max(7, len(df) * 0.75), 4.5))
    if error_col in df.columns:
        yerr = pd.to_numeric(df[error_col], errors="coerce").to_numpy(dtype=float)
        plt.bar(x, y, yerr=yerr, capsize=3)
    else:
        plt.bar(x, y)

    plt.xticks(x, [format_method_name(m) for m in df[method_col]], rotation=45, ha="right")
    plt.ylabel(metric)
    plt.xlabel("Uncertainty method")
    plt.title(title or f"{metric} by uncertainty method")
    plt.grid(axis="y", alpha=0.3)

    return save_current_figure(output_path, dpi=dpi, formats=formats)


def plot_timing_summary(
    timing_summary: pd.DataFrame,
    output_path: str | Path,
    value_col: str = "total_mean_s",
    error_col: str = "total_std_s",
    title: Optional[str] = None,
    dpi: int = 300,
    formats: Optional[str | Sequence[str]] = None,
    method_col: str = "method",
) -> dict[str, Path]:
    """Plot standalone timing by method."""
    timing_summary = _normalise_long_columns(timing_summary, method_col=method_col)
    required = {method_col, value_col}
    missing = required.difference(timing_summary.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = order_methods(timing_summary.copy(), method_col=method_col)
    x = np.arange(len(df))
    y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)

    plt.figure(figsize=(max(7, len(df) * 0.75), 4.5))
    if error_col in df.columns:
        yerr = pd.to_numeric(df[error_col], errors="coerce").to_numpy(dtype=float)
        plt.bar(x, y, yerr=yerr, capsize=3)
    else:
        plt.bar(x, y)

    plt.xticks(x, [format_method_name(m) for m in df[method_col]], rotation=45, ha="right")
    plt.ylabel("Time")
    plt.xlabel("Uncertainty method")
    plt.title(title or "Standalone total time by uncertainty method")
    plt.grid(axis="y", alpha=0.3)

    return save_current_figure(output_path, dpi=dpi, formats=formats)


def plot_ms_per_example(
    timing_summary: pd.DataFrame,
    output_path: str | Path,
    title: Optional[str] = None,
    dpi: int = 300,
    formats: Optional[str | Sequence[str]] = None,
    method_col: str = "method",
) -> dict[str, Path]:
    """Plot milliseconds per example by method."""
    return plot_timing_summary(
        timing_summary=timing_summary,
        output_path=output_path,
        value_col="ms_per_ex_mean",
        error_col="ms_per_ex_std",
        title=title or "Milliseconds per example by uncertainty method",
        dpi=dpi,
        formats=formats,
        method_col=method_col,
    )


def plot_metric_heatmap(
    metrics_summary: pd.DataFrame,
    output_path: str | Path,
    value_col: str = "mean",
    title: Optional[str] = None,
    dpi: int = 300,
    formats: Optional[str | Sequence[str]] = None,
    method_col: str = "method",
    metric_col: str = "metric",
) -> dict[str, Path]:
    """Plot a compact heatmap of metric means by method."""
    metrics_summary = _normalise_long_columns(metrics_summary, method_col=method_col, metric_col=metric_col)
    required = {method_col, metric_col, value_col}
    missing = required.difference(metrics_summary.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = metrics_summary[[method_col, metric_col, value_col]].copy()
    df = order_methods(order_metrics(df, metric_col=metric_col), method_col=method_col)
    matrix = df.pivot(index=method_col, columns=metric_col, values=value_col)

    metric_cols = [m for m in DEFAULT_METRIC_ORDER if m in matrix.columns]
    extra_cols = [c for c in matrix.columns if c not in metric_cols]
    matrix = matrix[metric_cols + extra_cols]

    plt.figure(figsize=(max(8, len(matrix.columns) * 0.9), max(4, len(matrix) * 0.45)))
    plt.imshow(matrix.values.astype(float), aspect="auto")

    plt.xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(matrix.index)), [format_method_name(m) for m in matrix.index])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            if pd.notna(value):
                plt.text(j, i, f"{float(value):.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(label=value_col)
    plt.title(title or "Metric summary heatmap")
    plt.xlabel("Metric")
    plt.ylabel("Method")

    return save_current_figure(output_path, dpi=dpi, formats=formats)


def plot_uncertainty_score_distribution(
    scores_long: pd.DataFrame,
    output_path: str | Path,
    methods: Optional[list[str]] = None,
    bins: int = 30,
    title: Optional[str] = None,
    dpi: int = 300,
    formats: Optional[str | Sequence[str]] = None,
    method_col: str = "method",
) -> dict[str, Path]:
    """Plot uncertainty score distributions by method."""
    scores_long = _normalise_long_columns(scores_long, method_col=method_col)
    required = {method_col, "uncertainty_score"}
    missing = required.difference(scores_long.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = scores_long.copy()
    if methods is not None:
        df = df[df[method_col].isin(methods)].copy()
    df = order_methods(df, method_col=method_col)

    method_list = list(dict.fromkeys(df[method_col].tolist()))
    if not method_list:
        raise ValueError("No methods available for score distribution plot.")

    plt.figure(figsize=(max(7, len(method_list) * 0.8), 4.5))
    for method in method_list:
        vals = pd.to_numeric(df[df[method_col] == method]["uncertainty_score"], errors="coerce").dropna().to_numpy(dtype=float)
        plt.hist(vals, bins=bins, alpha=0.35, label=format_method_name(method))

    plt.xlabel("Uncertainty score")
    plt.ylabel("Count")
    plt.title(title or "Uncertainty score distributions")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    return save_current_figure(output_path, dpi=dpi, formats=formats)


def make_all_figures(
    results_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    dpi: int = 300,
    formats: Optional[str | Sequence[str]] = None,
) -> dict:
    """Create standard figures from a completed results directory.

    This keeps the older one-call API but now tolerates evaluator outputs where
    method is called ``mode``.
    """
    formats = normalise_formats(formats)
    results_dir = Path(results_dir)
    output_dir = ensure_dir(output_dir or results_dir / "figures")

    metrics_summary_candidates = [
        results_dir / "metrics" / "metrics_summary_mean_std.csv",
        results_dir / "metrics_summary_mean_std.csv",
        results_dir / "metrics" / "metrics_summary_by_method.csv",
    ]
    timing_summary_candidates = [
        results_dir / "timing" / "method_total_times_summary_mean_std.csv",
        results_dir / "metrics" / "metric_times_all_folds.csv",
        results_dir / "metric_times_all_folds.csv",
    ]

    metrics_summary_path = next((p for p in metrics_summary_candidates if p.exists()), None)
    if metrics_summary_path is None:
        raise FileNotFoundError(
            "Metrics summary not found. Checked: " + ", ".join(str(p) for p in metrics_summary_candidates)
        )

    metrics_summary = pd.read_csv(metrics_summary_path)
    metrics_summary = _normalise_long_columns(metrics_summary)
    outputs: dict = {}

    for metric in ["ECE", "AU-PRC", "ROC-AUC", "TI@95"]:
        if "metric" in metrics_summary.columns and metric in set(metrics_summary["metric"]):
            outputs[f"metric_{metric}"] = plot_metric_summary(
                metrics_summary,
                metric=metric,
                output_path=output_dir / f"metric_{_safe_name(metric)}.png",
                dpi=dpi,
                formats=formats,
            )

    outputs["metric_heatmap"] = plot_metric_heatmap(
        metrics_summary,
        output_path=output_dir / "metric_heatmap.png",
        dpi=dpi,
        formats=formats,
    )

    timing_summary_path = next((p for p in timing_summary_candidates if p.exists()), None)
    if timing_summary_path is not None:
        timing_summary = pd.read_csv(timing_summary_path)
        timing_summary = _normalise_long_columns(timing_summary)

        if {"method", "total_mean_s"}.issubset(timing_summary.columns):
            outputs["timing_total"] = plot_timing_summary(
                timing_summary,
                output_path=output_dir / "timing_total_seconds.png",
                dpi=dpi,
                formats=formats,
            )

        if {"method", "ms_per_ex_mean"}.issubset(timing_summary.columns):
            outputs["timing_ms_per_example"] = plot_ms_per_example(
                timing_summary,
                output_path=output_dir / "timing_ms_per_example.png",
                dpi=dpi,
                formats=formats,
            )

    score_files = sorted((results_dir / "scores").glob("*_scores_long.csv"))
    if score_files:
        scores_long = pd.concat([pd.read_csv(path) for path in score_files], ignore_index=True)
        outputs["score_distributions"] = plot_uncertainty_score_distribution(
            scores_long,
            output_path=output_dir / "uncertainty_score_distributions.png",
            dpi=dpi,
            formats=formats,
        )

    return outputs


__all__ = [
    "DEFAULT_METHOD_ORDER",
    "DEFAULT_METRIC_ORDER",
    "METHOD_DISPLAY_NAMES",
    "ensure_parent_dir",
    "ensure_dir",
    "normalise_formats",
    "maybe_savefig",
    "save_current_figure",
    "format_method_name",
    "order_methods",
    "order_metrics",
    "plot_correlation_heatmap",
    "save_correlation_heatmaps",
    "plot_risk_coverage_curve",
    "plot_multiple_risk_coverage_curves",
    "plot_rejection_metric_curve",
    "plot_macro_delta_curve",
    "plot_pct_incorrect_rejected_curve",
    "plot_reliability_diagram",
    "plot_metric_by_method",
    "plot_metric_summary",
    "plot_timing_summary",
    "plot_ms_per_example",
    "plot_metric_heatmap",
    "plot_uncertainty_score_distribution",
    "make_all_figures",
]
