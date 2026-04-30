#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_fold_metrics.py

Aggregate fold-level uncertainty metric summary files and produce:
  1. A LaTeX table of mean ± standard deviation across folds.
  2. A CSV file with best-vs-other statistical comparisons.

The LaTeX table uses:
  - bold = best mean for that metric
  - underline = statistically indistinguishable from the best method, p >= alpha

Best-method rules:
  - Higher is better: ROC-AUC, AU-PRC, Norm RC-AUC, TI, TI@95
  - C-Slope: closest to 1
  - CITL: closest to 0
  - All other metrics: lower is better

Example
-------
python scripts/summarize_fold_metrics.py \
  --summary-glob "results/fold_metrics/ru_fold*_metrics_summary*.csv" \
  --lang RU \
  --outdir results/tables
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


# ---------------------------------------------------------------------
# Metric and method configuration
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
# Loading and cleaning
# ---------------------------------------------------------------------

def normalise_metric_name(name: str) -> str:
    """Normalise selected aliases while preserving readable metric names."""
    s = str(name).strip()
    low = s.lower()

    aliases = {
        "roc-auc": "ROC-AUC",
        "n.rc-auc": "Norm RC-AUC",
        "nrc-auc": "Norm RC-AUC",
        "norm rc-auc": "Norm RC-AUC",
        "e-auopt rc": "E-AUoptRC",
        "e-auopt": "E-AUoptRC",
        "calibration slope": "C-Slope",
        "slope": "C-Slope",
        "cal-in-the-large": "CITL",
        "calibration-in-the-large": "CITL",
        "expected calibration error": "ECE",
        "ti-95": "TI@95",
        "ti @95": "TI@95",
    }
    return aliases.get(low, s)


def load_fold_summary(path: str | Path) -> pd.DataFrame:
    """Load one fold-level metric summary CSV.

    Expected format:
      - rows = metrics
      - columns = uncertainty methods
    """
    df = pd.read_csv(path, index_col=0)
    df.index = [normalise_metric_name(x) for x in df.index]

    # If a metric appears more than once after alias normalisation, keep the first.
    # This avoids duplicated rows such as ROC-AUC and roc-auc.
    df = df[~pd.Index(df.index).duplicated(keep="first")]

    # Convert all values to numeric; non-numeric values become NaN.
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_all_folds(summary_glob: str) -> Tuple[List[Path], List[pd.DataFrame]]:
    files = sorted(Path(p) for p in glob.glob(summary_glob))
    if not files:
        raise FileNotFoundError(f"No fold summary files matched: {summary_glob}")

    fold_dfs = [load_fold_summary(p) for p in files]
    return files, fold_dfs


def common_methods(fold_dfs: Sequence[pd.DataFrame]) -> List[str]:
    """Return methods present in every fold, preserving first-fold order."""
    if not fold_dfs:
        return []
    first = list(fold_dfs[0].columns)
    return [m for m in first if all(m in df.columns for df in fold_dfs)]


def common_metrics(fold_dfs: Sequence[pd.DataFrame]) -> List[str]:
    """Return metrics present in every fold, preserving first-fold order."""
    if not fold_dfs:
        return []
    first = list(fold_dfs[0].index)
    return [m for m in first if all(m in df.index for df in fold_dfs)]


def build_stacked_frame(fold_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Build a MultiIndex DataFrame with index = (fold, metric)."""
    methods = common_methods(fold_dfs)
    metrics = common_metrics(fold_dfs)

    if len(methods) < 2:
        raise ValueError("Need at least two common methods across folds.")
    if not metrics:
        raise ValueError("No common metrics found across folds.")

    cleaned = [df.loc[metrics, methods].copy() for df in fold_dfs]
    return pd.concat(cleaned, keys=range(len(cleaned)), names=["fold", "metric"])


# ---------------------------------------------------------------------
# Best method selection and statistical tests
# ---------------------------------------------------------------------

def choose_best_method(metric: str, mean_row: pd.Series) -> str:
    """Choose the best method for a metric using the benchmark conventions."""
    valid = mean_row.dropna()
    if valid.empty:
        return ""

    if metric in HIGHER_IS_BETTER:
        return str(valid.idxmax())
    if metric == "C-Slope":
        return str((valid - 1.0).abs().idxmin())
    if metric == "CITL":
        return str(valid.abs().idxmin())
    return str(valid.idxmin())


def paired_test(best_values: pd.Series, other_values: pd.Series, alpha: float) -> Tuple[str, float, float]:
    """Run a paired test between the best method and one other method.

    Returns:
      test_name, normality_p, p_value
    """
    paired = pd.concat([best_values, other_values], axis=1).dropna()
    if paired.shape[0] < 2:
        return "N/A", np.nan, np.nan

    x = paired.iloc[:, 0].astype(float)
    y = paired.iloc[:, 1].astype(float)
    diffs = x - y

    if np.allclose(diffs.values, 0.0, atol=0.0, rtol=0.0):
        return "No difference", np.nan, 1.0

    normality_p = np.nan
    if len(diffs) >= 3:
        try:
            normality_p = float(shapiro(diffs).pvalue)
        except Exception:
            normality_p = np.nan

    try:
        if np.isfinite(normality_p) and normality_p >= alpha:
            _, p_value = ttest_rel(x, y, nan_policy="omit")
            return "Paired t-test", normality_p, float(p_value)

        _, p_value = wilcoxon(x, y, zero_method="zsplit", alternative="two-sided", mode="auto")
        return "Wilcoxon", normality_p, float(p_value)
    except Exception:
        return "N/A", normality_p, np.nan


def compute_summary_and_tests(
    stacked: pd.DataFrame,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], Dict[str, List[str]]]:
    """Compute mean/std, best method, and best-vs-other tests."""
    mean_vals = stacked.groupby(level="metric").mean().astype(float)
    std_vals = stacked.groupby(level="metric").std().astype(float)

    metrics = list(mean_vals.index)
    methods = list(mean_vals.columns)

    best_method: Dict[str, str] = {}
    close_methods: Dict[str, List[str]] = {m: [] for m in metrics}
    records = []

    for metric in metrics:
        best = choose_best_method(metric, mean_vals.loc[metric])
        best_method[metric] = best

        if not best:
            continue

        metric_df = stacked.xs(metric, level="metric").astype(float)
        for other in methods:
            if other == best:
                continue

            test_name, normality_p, p_value = paired_test(metric_df[best], metric_df[other], alpha=alpha)

            records.append(
                {
                    "metric": metric,
                    "best_method": best,
                    "other_method": other,
                    "test": test_name,
                    "normality_p": normality_p,
                    "p_value": p_value,
                    "not_significantly_different_from_best": bool(np.isfinite(p_value) and p_value >= alpha),
                }
            )

            if np.isfinite(p_value) and p_value >= alpha:
                close_methods[metric].append(other)

    pairwise_df = pd.DataFrame.from_records(records)
    summary = pd.concat({"mean": mean_vals, "std": std_vals}, axis=1)
    return summary, pairwise_df, best_method, close_methods


# ---------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------

def ordered_methods(methods: Iterable[str]) -> List[str]:
    methods = list(methods)
    ordered = [m for m in DEFAULT_METHOD_ORDER if m in methods]
    ordered.extend([m for m in methods if m not in ordered])
    return ordered


def ordered_metrics(metrics: Iterable[str]) -> List[Tuple[str, List[str]]]:
    """Return metric groups using known groups, then an 'Other Metrics' group."""
    available = list(metrics)
    seen = set()
    groups = []

    for group_name, group_metrics in METRIC_GROUPS.items():
        selected = [m for m in group_metrics if m in available]
        if selected:
            groups.append((group_name, selected))
            seen.update(selected)

    other = [m for m in available if m not in seen]
    if other:
        groups.append(("Other Metrics", other))

    return groups


def latex_escape(text: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def format_mean(value: float, method: str, best: str, close: Sequence[str]) -> str:
    if pd.isna(value):
        return "--"

    s = f"{value:.2f}"
    if method == best:
        return rf"\textbf{{{s}}}"
    if method in set(close):
        return rf"\underline{{{s}}}"
    return s


def format_std(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.3f}"


def build_latex_table(
    summary: pd.DataFrame,
    best_method: Dict[str, str],
    close_methods: Dict[str, List[str]],
    lang: str,
    alpha: float,
    label: str | None = None,
) -> str:
    """Build a grouped LaTeX table."""
    mean_vals = summary["mean"]
    std_vals = summary["std"]
    methods = ordered_methods(mean_vals.columns)
    metric_groups = ordered_metrics(mean_vals.index)

    if label is None:
        label = f"tab:metrics_summary_{lang.lower()}"

    colspec = "l|" + "|".join([r"r@{\hspace{0.2cm}}r" for _ in methods])

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    method_headers = [
        rf"\multicolumn{{2}}{{c}}{{{latex_escape(METHOD_DISPLAY_NAMES.get(m, m))}}}"
        for m in methods
    ]
    lines.append("Metric & " + " & ".join(method_headers) + r" \\")
    lines.append(" & " + " & ".join(["Mean & STD" for _ in methods]) + r" \\")
    lines.append(r"\midrule")

    for group_name, group_metrics in metric_groups:
        lines.append(
            rf"\multicolumn{{{1 + 2 * len(methods)}}}{{l}}{{\textbf{{{latex_escape(group_name)}}}}} \\"
        )
        lines.append(r"\midrule")

        for metric in group_metrics:
            display_metric = METRIC_DISPLAY_NAMES.get(metric, metric)
            row = [latex_escape(display_metric)]
            best = best_method.get(metric, "")
            close = close_methods.get(metric, [])

            for method in methods:
                mu = mean_vals.loc[metric, method]
                sd = std_vals.loc[metric, method]
                row.append(format_mean(mu, method, best, close))
                row.append(format_std(sd))

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(
        rf"\caption{{{latex_escape(lang)} fold-level metric summary. Values are mean and standard deviation across folds. "
        rf"\textbf{{Bold}} marks the best method for each metric. "
        rf"\underline{{Underlined}} methods are not significantly different from the best method "
        rf"according to paired tests at $\alpha={alpha:.2f}$.}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise fold-level metric CSVs and generate LaTeX tables."
    )
    parser.add_argument(
        "--summary-glob",
        required=True,
        help="Glob pattern for fold-level metric summary CSV files, e.g. 'results/fold_metrics/ru_fold*_metrics_summary*.csv'.",
    )
    parser.add_argument(
        "--lang",
        required=True,
        help="Language code or name used in output filenames and captions, e.g. RU or Arabic.",
    )
    parser.add_argument(
        "--outdir",
        default="results/tables",
        help="Output directory for LaTeX and CSV files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for paired tests. Default: 0.05.",
    )
    parser.add_argument(
        "--latex-label",
        default=None,
        help="Optional custom LaTeX label. Default: tab:metrics_summary_<lang>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files, fold_dfs = load_all_folds(args.summary_glob)
    print(f"[info] Found {len(files)} fold summary file(s).")
    for p in files:
        print(f"       - {p}")

    stacked = build_stacked_frame(fold_dfs)
    summary, pairwise_df, best_method, close_methods = compute_summary_and_tests(
        stacked,
        alpha=args.alpha,
    )

    lang_slug = args.lang.lower().replace(" ", "_")

    pairwise_path = outdir / f"{lang_slug}_pairwise_best_vs_others.csv"
    pairwise_df.to_csv(pairwise_path, index=False)

    latex = build_latex_table(
        summary=summary,
        best_method=best_method,
        close_methods=close_methods,
        lang=args.lang,
        alpha=args.alpha,
        label=args.latex_label,
    )

    latex_path = outdir / f"{lang_slug}_metrics_table.tex"
    latex_path.write_text(latex, encoding="utf-8")

    summary_path = outdir / f"{lang_slug}_metric_summary_mean_std.csv"
    summary.to_csv(summary_path)

    print(f"[ok] Wrote LaTeX table: {latex_path}")
    print(f"[ok] Wrote pairwise tests: {pairwise_path}")
    print(f"[ok] Wrote mean/std summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
