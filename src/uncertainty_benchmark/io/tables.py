"""Paper-ready table utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METHOD_DISPLAY_NAMES = {
    "SR": "Softmax Response",
    "ENT": "Predictive Entropy",
    "SMP": "Sampled Max Probability",
    "PV": "Probability Variance",
    "BALD": "BALD",
    "ENT_MC": "MC Predictive Entropy",
    "MD": "Mahalanobis Distance",
    "HUQ-MD": "HUQ-MD",
    "LOF": "Local Outlier Factor",
    "ISOF": "Isolation Forest",
}


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
    "E-AUoptRC",
    "TI",
    "TI@95",
    "Optimal Coverage",
]


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_mean_std(mean, std, decimals: int = 3) -> str:
    """Format mean ± standard deviation."""
    if pd.isna(mean):
        return ""

    if pd.isna(std):
        return f"{mean:.{decimals}f}"

    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def add_method_display_names(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    """Add readable method names."""
    out = df.copy()
    out["method_name"] = out[method_col].map(METHOD_DISPLAY_NAMES).fillna(out[method_col])
    return out


def order_methods(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    """Sort by the standard method order."""
    out = df.copy()
    order_map = {method: i for i, method in enumerate(DEFAULT_METHOD_ORDER)}
    out["_method_order"] = out[method_col].map(order_map).fillna(999)
    out = out.sort_values(["_method_order", method_col]).drop(columns=["_method_order"])
    return out


def order_metrics(df: pd.DataFrame, metric_col: str = "metric") -> pd.DataFrame:
    """Sort by the standard metric order."""
    out = df.copy()
    order_map = {metric: i for i, metric in enumerate(DEFAULT_METRIC_ORDER)}
    out["_metric_order"] = out[metric_col].map(order_map).fillna(999)
    out = out.sort_values(["_metric_order", metric_col]).drop(columns=["_metric_order"])
    return out


def build_metrics_paper_table(
    metrics_summary: pd.DataFrame,
    decimals: int = 3,
) -> pd.DataFrame:
    """Create a wide paper table from metrics_summary_mean_std.csv.

    Expected input columns:
        method, metric, mean, std

    Output:
        one row per method, one column per metric.
    """
    required = {"method", "metric", "mean", "std"}
    missing = required.difference(metrics_summary.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    rows = []

    for _, row in metrics_summary.iterrows():
        rows.append(
            {
                "method": row["method"],
                "metric": row["metric"],
                "value": format_mean_std(row["mean"], row["std"], decimals=decimals),
            }
        )

    tidy = pd.DataFrame(rows)
    tidy = order_methods(order_metrics(tidy))

    wide = tidy.pivot(index="method", columns="metric", values="value").reset_index()

    # Reorder metric columns
    cols = ["method"] + [m for m in DEFAULT_METRIC_ORDER if m in wide.columns]
    extra_cols = [c for c in wide.columns if c not in cols]
    wide = wide[cols + extra_cols]

    wide = add_method_display_names(wide)
    wide = wide.rename(columns={"method": "Method ID", "method_name": "Method"})

    # Put readable name first
    cols = ["Method", "Method ID"] + [
        c for c in wide.columns if c not in {"Method", "Method ID"}
    ]

    return wide[cols]


def build_timing_paper_table(
    total_times_summary: pd.DataFrame,
    decimals: int = 3,
) -> pd.DataFrame:
    """Create a paper timing table from method_total_times_summary_mean_std.csv.

    Expected input columns:
        method,
        uncertainty_mean_s,
        uncertainty_std_s,
        metrics_mean_s,
        metrics_std_s,
        total_mean_s,
        total_std_s,
        ms_per_ex_mean,
        ms_per_ex_std,
        ex_per_s_mean
    """
    required = {
        "method",
        "uncertainty_mean_s",
        "uncertainty_std_s",
        "metrics_mean_s",
        "metrics_std_s",
        "total_mean_s",
        "total_std_s",
        "ms_per_ex_mean",
        "ms_per_ex_std",
        "ex_per_s_mean",
    }
    missing = required.difference(total_times_summary.columns)

    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = order_methods(total_times_summary.copy())
    df = add_method_display_names(df)

    out = pd.DataFrame(
        {
            "Method": df["method_name"],
            "Method ID": df["method"],
            "Uncertainty time (s)": [
                format_mean_std(m, s, decimals)
                for m, s in zip(df["uncertainty_mean_s"], df["uncertainty_std_s"])
            ],
            "Metric time (s)": [
                format_mean_std(m, s, decimals)
                for m, s in zip(df["metrics_mean_s"], df["metrics_std_s"])
            ],
            "Total time (s)": [
                format_mean_std(m, s, decimals)
                for m, s in zip(df["total_mean_s"], df["total_std_s"])
            ],
            "ms / example": [
                format_mean_std(m, s, decimals)
                for m, s in zip(df["ms_per_ex_mean"], df["ms_per_ex_std"])
            ],
            "examples / s": [
                "" if pd.isna(v) else f"{v:.{decimals}f}"
                for v in df["ex_per_s_mean"]
            ],
        }
    )

    return out


def save_latex_table(
    df: pd.DataFrame,
    path,
    caption: str,
    label: str,
    index: bool = False,
    longtable: bool = False,
) -> Path:
    """Save a dataframe as a LaTeX table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    latex = df.to_latex(
        index=index,
        escape=True,
        caption=caption,
        label=label,
        longtable=longtable,
    )

    path.write_text(latex, encoding="utf-8")
    return path


def create_paper_tables(
    results_dir,
    output_dir=None,
    decimals: int = 3,
) -> dict:
    """Create paper-ready metric and timing tables from a results directory.

    Expected files:
        results_dir/metrics/metrics_summary_mean_std.csv
        results_dir/timing/method_total_times_summary_mean_std.csv

    Outputs:
        metrics_summary_paper.csv
        metrics_summary_paper.tex
        timing_summary_paper.csv
        timing_summary_paper.tex
    """
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = results_dir / "paper_tables"

    output_dir = ensure_dir(output_dir)

    metrics_summary_path = results_dir / "metrics" / "metrics_summary_mean_std.csv"
    timing_summary_path = (
        results_dir / "timing" / "method_total_times_summary_mean_std.csv"
    )

    if not metrics_summary_path.exists():
        raise FileNotFoundError(f"Metrics summary not found: {metrics_summary_path}")

    if not timing_summary_path.exists():
        raise FileNotFoundError(f"Timing summary not found: {timing_summary_path}")

    metrics_summary = pd.read_csv(metrics_summary_path)
    timing_summary = pd.read_csv(timing_summary_path)

    metrics_table = build_metrics_paper_table(
        metrics_summary,
        decimals=decimals,
    )
    timing_table = build_timing_paper_table(
        timing_summary,
        decimals=decimals,
    )

    metrics_csv = output_dir / "metrics_summary_paper.csv"
    metrics_tex = output_dir / "metrics_summary_paper.tex"

    timing_csv = output_dir / "timing_summary_paper.csv"
    timing_tex = output_dir / "timing_summary_paper.tex"

    metrics_table.to_csv(metrics_csv, index=False)
    timing_table.to_csv(timing_csv, index=False)

    save_latex_table(
        metrics_table,
        metrics_tex,
        caption=(
            "Mean and standard deviation of uncertainty evaluation metrics "
            "across folds."
        ),
        label="tab:uncertainty-metrics",
        index=False,
    )

    save_latex_table(
        timing_table,
        timing_tex,
        caption=(
            "Standalone uncertainty scoring and metric-computation time "
            "by method."
        ),
        label="tab:uncertainty-timing",
        index=False,
    )

    return {
        "metrics_csv": metrics_csv,
        "metrics_tex": metrics_tex,
        "timing_csv": timing_csv,
        "timing_tex": timing_tex,
    }
