#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uncertainty_benchmark.reporting.latex_tables

Reusable LaTeX table builders for uncertainty benchmark reporting.

This module keeps table-generation logic separate from metric computation and
analysis. It is designed to support the scripts:

    scripts/summarize_fold_metrics.py
    scripts/analyze_metric_correlations.py
    scripts/analyze_rejection_f1.py

Main table types
----------------
1. Fold-level metric summary table
   - mean/std across folds
   - bold best method
   - underline methods statistically tied with best

2. Metric-correlation tables
   - Kendall tau with significance styling
   - Pearson r

3. Rejection table
   - Macro ΔF1 and % Incorrect among rejected examples
   - bold best methods
   - underline statistically/practically tied methods
   - dagger significance markers

The functions return LaTeX strings. File writing should normally be handled by
scripts or caller code.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Display defaults
# ---------------------------------------------------------------------

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
        "Optimal Coverage",
    ],
}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def latex_escape(text: object) -> str:
    """Escape common LaTeX special characters in plain text."""
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


def ordered_methods(
    methods: Iterable[str],
    preferred_order: Sequence[str] = DEFAULT_METHOD_ORDER,
) -> List[str]:
    """Order methods according to a preferred order and append unknown methods."""
    method_list = list(methods)
    ordered = [m for m in preferred_order if m in method_list]
    ordered.extend([m for m in method_list if m not in ordered])
    return ordered


def group_metrics(
    metrics: Iterable[str],
    metric_groups: Mapping[str, Sequence[str]] = METRIC_GROUPS,
) -> List[Tuple[str, List[str]]]:
    """Group metrics into known groups plus 'Other Metrics'."""
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


def display_method(method: str, method_display_names: Mapping[str, str] = METHOD_DISPLAY_NAMES) -> str:
    """Return display label for a method."""
    return method_display_names.get(method, method)


def display_metric(metric: str, metric_display_names: Mapping[str, str] = METRIC_DISPLAY_NAMES) -> str:
    """Return display label for a metric."""
    return metric_display_names.get(metric, metric)


def format_float(value: object, decimals: int = 2, missing: str = "--") -> str:
    """Format a numeric value for LaTeX."""
    try:
        if pd.isna(value):
            return missing
        return f"{float(value):.{decimals}f}"
    except Exception:
        return missing


def style_text(
    text: str,
    bold: bool = False,
    underline: bool = False,
) -> str:
    """Apply simple LaTeX styling to text."""
    if bold:
        return rf"\textbf{{{text}}}"
    if underline:
        return rf"\underline{{{text}}}"
    return text


def format_mean_with_style(
    value: object,
    is_best: bool = False,
    is_tied: bool = False,
    decimals: int = 2,
    missing: str = "--",
) -> str:
    """Format a mean value and apply best/tied styling."""
    text = format_float(value, decimals=decimals, missing=missing)
    if text == missing:
        return text
    return style_text(text, bold=is_best, underline=(not is_best and is_tied))


# ---------------------------------------------------------------------
# Fold-level metric summary table
# ---------------------------------------------------------------------

def build_metric_summary_table(
    mean_values: pd.DataFrame,
    std_values: pd.DataFrame,
    best_methods: Mapping[str, str],
    close_methods: Optional[Mapping[str, Sequence[str]]] = None,
    lang: str = "",
    alpha: float = 0.05,
    label: Optional[str] = None,
    caption: Optional[str] = None,
    resize_to_textwidth: bool = True,
    mean_decimals: int = 2,
    std_decimals: int = 3,
) -> str:
    """Build a LaTeX table for fold-level metric summaries.

    Parameters
    ----------
    mean_values:
        DataFrame with metrics as rows and methods as columns.
    std_values:
        DataFrame with metrics as rows and methods as columns.
    best_methods:
        Mapping from metric to best method.
    close_methods:
        Mapping from metric to methods not significantly different from the best.
    lang:
        Language code/name used in caption and label.
    alpha:
        Significance threshold used for underlining.
    """
    close_methods = dict(close_methods or {})
    methods = ordered_methods(mean_values.columns)
    metric_groups = group_metrics(mean_values.index)

    if label is None:
        label = f"tab:metrics_summary_{str(lang).lower().replace(' ', '_')}" if lang else "tab:metrics_summary"

    if caption is None:
        prefix = f"{latex_escape(lang)} fold-level metric summary" if lang else "Fold-level metric summary"
        caption = (
            f"{prefix}. Values are mean and standard deviation across folds. "
            r"\textbf{Bold} marks the best method for each metric. "
            rf"\underline{{Underlined}} methods are not significantly different from the best method at $\alpha={alpha:.2f}$."
        )

    colspec = "l|" + "|".join([r"r@{\hspace{0.2cm}}r" for _ in methods])

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    if resize_to_textwidth:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    method_headers = [
        rf"\multicolumn{{2}}{{c}}{{{latex_escape(display_method(method))}}}"
        for method in methods
    ]
    lines.append("Metric & " + " & ".join(method_headers) + r" \\")
    lines.append(" & " + " & ".join(["Mean & STD" for _ in methods]) + r" \\")
    lines.append(r"\midrule")

    for group_name, metrics in metric_groups:
        lines.append(
            rf"\multicolumn{{{1 + 2 * len(methods)}}}{{l}}{{\textbf{{{latex_escape(group_name)}}}}} \\"
        )
        lines.append(r"\midrule")

        for metric in metrics:
            row = [latex_escape(display_metric(str(metric)))]
            best = best_methods.get(str(metric), "")
            tied = set(close_methods.get(str(metric), []))

            for method in methods:
                mean_text = format_mean_with_style(
                    mean_values.loc[metric, method],
                    is_best=(method == best),
                    is_tied=(method in tied),
                    decimals=mean_decimals,
                )
                std_text = format_float(std_values.loc[metric, method], decimals=std_decimals)
                row.extend([mean_text, std_text])

            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    if resize_to_textwidth:
        lines.append(r"}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_metric_summary_table_from_wide_summary(
    summary: pd.DataFrame,
    best_methods: Mapping[str, str],
    close_methods: Optional[Mapping[str, Sequence[str]]] = None,
    **kwargs,
) -> str:
    """Build metric summary table from a wide summary with top columns mean/std."""
    if "mean" not in summary.columns.get_level_values(0) or "std" not in summary.columns.get_level_values(0):
        raise ValueError("summary must have top-level columns 'mean' and 'std'.")

    return build_metric_summary_table(
        mean_values=summary["mean"],
        std_values=summary["std"],
        best_methods=best_methods,
        close_methods=close_methods,
        **kwargs,
    )


# ---------------------------------------------------------------------
# Correlation tables
# ---------------------------------------------------------------------

def format_tau_with_p(tau: object, p_value: object) -> str:
    """Format Kendall tau with significance styling.

    - bold if p < 0.01
    - underline if p < 0.05
    """
    if pd.isna(tau):
        return "--"

    text = f"{float(tau):.2f}"
    try:
        p = float(p_value)
    except Exception:
        p = np.nan

    if np.isfinite(p):
        if p < 0.01:
            return rf"\textbf{{{text}}}"
        if p < 0.05:
            return rf"\underline{{{text}}}"
    return text


def build_kendall_correlation_table(
    tau_tables: Mapping[str, pd.DataFrame],
    tau_p_tables: Mapping[str, pd.DataFrame],
    caption: Optional[str] = None,
    label: str = "tab:metric_correlations_kendall",
    table_star: bool = True,
) -> str:
    """Build a LaTeX table for Kendall tau metric correlations."""
    if not tau_tables:
        raise ValueError("tau_tables is empty.")

    first_group = next(iter(tau_tables))
    langs = list(tau_tables[first_group].columns)

    if caption is None:
        caption = (
            r"Kendall's correlation ($\tau$) between uncertainty metric pairs, computed by "
            r"concatenating method-level values across folds for each language. "
            r"Bold indicates $p<0.01$; underline indicates $p<0.05$."
        )

    env = "table*" if table_star else "table"
    lines: List[str] = []
    lines.append(rf"\begin{{{env}}}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l " + " ".join(["r"] * len(langs)) + r"}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Metric Pair} & "
        + " & ".join(rf"\textbf{{{latex_escape(lang)}}}" for lang in langs)
        + r" \\"
    )
    lines.append(r"\midrule")

    for group, table in tau_tables.items():
        p_table = tau_p_tables[group]
        lines.append(rf"\multicolumn{{{len(langs) + 1}}}{{l}}{{\textbf{{{latex_escape(group)}}}}} \\")
        lines.append(r"\midrule")

        for metric_1, metric_2 in table.index:
            row = [latex_escape(f"{metric_1} vs {metric_2}")]
            for lang in langs:
                row.append(format_tau_with_p(table.loc[(metric_1, metric_2), lang], p_table.loc[(metric_1, metric_2), lang]))
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


def build_pearson_correlation_table(
    r_tables: Mapping[str, pd.DataFrame],
    caption: Optional[str] = None,
    label: str = "tab:metric_correlations_pearson",
    table_star: bool = True,
) -> str:
    """Build a LaTeX table for Pearson r metric correlations."""
    if not r_tables:
        raise ValueError("r_tables is empty.")

    first_group = next(iter(r_tables))
    langs = list(r_tables[first_group].columns)

    if caption is None:
        caption = (
            r"Pearson correlation ($r$) between uncertainty metric pairs, computed by "
            r"concatenating method-level values across folds for each language."
        )

    env = "table*" if table_star else "table"
    lines: List[str] = []
    lines.append(rf"\begin{{{env}}}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{l " + " ".join(["r"] * len(langs)) + r"}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Metric Pair} & "
        + " & ".join(rf"\textbf{{{latex_escape(lang)}}}" for lang in langs)
        + r" \\"
    )
    lines.append(r"\midrule")

    for group, table in r_tables.items():
        lines.append(rf"\multicolumn{{{len(langs) + 1}}}{{l}}{{\textbf{{{latex_escape(group)}}}}} \\")
        lines.append(r"\midrule")

        for metric_1, metric_2 in table.index:
            row = [latex_escape(f"{metric_1} vs {metric_2}")]
            for lang in langs:
                row.append(format_float(table.loc[(metric_1, metric_2), lang], decimals=2))
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Rejection table
# ---------------------------------------------------------------------

def format_delta_mean(value: object) -> str:
    """Format Macro ΔF1 mean in percentage points."""
    try:
        if pd.isna(value):
            return "--"
        value = float(value)
        if abs(value) < 0.005:
            return "0"
        return f"{value:.2f}"
    except Exception:
        return "--"


def format_delta_std(value: object) -> str:
    return format_float(value, decimals=2)


def format_pct_mean(value: object) -> str:
    return format_float(value, decimals=1)


def format_pct_std(value: object) -> str:
    return format_float(value, decimals=1)


def add_dagger_if_significant(text: str, p_value: object, alpha: float = 0.05) -> str:
    """Append dagger if p-value is below alpha."""
    try:
        p = float(p_value)
    except Exception:
        return text

    if np.isfinite(p) and p < alpha:
        return text + r"$^{\dagger}$"
    return text


def style_rejection_mean(
    text: str,
    is_best: bool,
    is_tied: bool,
    p_value: object,
    alpha: float = 0.05,
) -> str:
    """Style rejection mean cell with dagger/bold/underline."""
    if text == "--":
        return text

    text = add_dagger_if_significant(text, p_value, alpha=alpha)

    if is_best:
        return rf"\textbf{{{text}}}"
    if is_tied:
        return rf"\underline{{{text}}}"
    return text


def write_rejection_block(
    lines: List[str],
    aggregate: Mapping[str, object],
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    block_title: Optional[str] = None,
    significance_alpha: float = 0.05,
) -> None:
    """Append one rejection-table block to a list of LaTeX lines.

    Expected aggregate keys:
      mean_macro, std_macro, p_macro,
      mean_pct, std_pct, p_pct,
      best_macro_idx, tied_macro_mask,
      best_pct_idx, tied_pct_mask
    """
    n_rates = len(rejection_rates)
    col_format = "l " + " ".join(
        [r"@{\hspace{0.25cm}}r@{\hspace{0.1cm}}r@{\hspace{0.2cm}}r@{\hspace{0.1cm}}r" for _ in rejection_rates]
    )

    lines.append(rf"\begin{{tabular}}{{{col_format}}}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join([rf"\multicolumn{{4}}{{c}}{{{int(rate * 100)}\%}}" for rate in rejection_rates]) + r" \\")
    lines.append(
        r"\multirow{2}{*}{Method} "
        + "".join([r" & \multicolumn{2}{c}{Macro $\Delta$F1} & \multicolumn{2}{c}{\% Incorrect}" for _ in rejection_rates])
        + r" \\"
    )
    lines.append(" " + (r"&$\mu$&$\sigma$&$\mu$&$\sigma$" * n_rates) + r"\\")
    lines.append(r"\midrule")

    if block_title:
        lines.append(rf"\multicolumn{{{1 + 4 * n_rates}}}{{l}}{{\textbf{{{latex_escape(block_title)}}}}}\\")
        lines.append(r"\midrule")

    mean_macro = np.asarray(aggregate["mean_macro"], dtype=float)
    std_macro = np.asarray(aggregate["std_macro"], dtype=float)
    mean_pct = np.asarray(aggregate["mean_pct"], dtype=float)
    std_pct = np.asarray(aggregate["std_pct"], dtype=float)
    p_macro = np.asarray(aggregate["p_macro"], dtype=float)
    p_pct = np.asarray(aggregate["p_pct"], dtype=float)
    best_macro_idx = np.asarray(aggregate["best_macro_idx"], dtype=int)
    tied_macro_mask = np.asarray(aggregate["tied_macro_mask"], dtype=bool)
    best_pct_idx = np.asarray(aggregate["best_pct_idx"], dtype=int)
    tied_pct_mask = np.asarray(aggregate["tied_pct_mask"], dtype=bool)

    for method_idx, method in enumerate(methods):
        cells = [latex_escape(display_method(method))]

        for rate_idx in range(n_rates):
            macro_mu = style_rejection_mean(
                format_delta_mean(mean_macro[method_idx, rate_idx]),
                is_best=(method_idx == best_macro_idx[rate_idx]),
                is_tied=(method_idx != best_macro_idx[rate_idx] and tied_macro_mask[method_idx, rate_idx]),
                p_value=p_macro[method_idx, rate_idx],
                alpha=significance_alpha,
            )
            macro_sd = format_delta_std(std_macro[method_idx, rate_idx])
            cells.extend([macro_mu, macro_sd])

            pct_mu = style_rejection_mean(
                format_pct_mean(mean_pct[method_idx, rate_idx]),
                is_best=(method_idx == best_pct_idx[rate_idx]),
                is_tied=(method_idx != best_pct_idx[rate_idx] and tied_pct_mask[method_idx, rate_idx]),
                p_value=p_pct[method_idx, rate_idx],
                alpha=significance_alpha,
            )
            pct_sd = format_pct_std(std_pct[method_idx, rate_idx])
            cells.extend([pct_mu, pct_sd])

        lines.append(" " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")


def build_rejection_table(
    aggregates: Mapping[str, Mapping[str, object]],
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    lang: str = "",
    alpha_tie: float = 0.05,
    macro_tolerance_pp: float = 0.10,
    pct_tolerance_pp: float = 1.00,
    significance_alpha: float = 0.05,
    label: Optional[str] = None,
    caption: Optional[str] = None,
) -> str:
    """Build full LaTeX rejection table."""
    if label is None:
        label = f"tab:rejection_f1_{str(lang).lower().replace(' ', '_')}" if lang else "tab:rejection_f1"

    if caption is None:
        prefix = f"{latex_escape(lang)} uncertainty-guided rejection results" if lang else "Uncertainty-guided rejection results"
        caption = (
            rf"{prefix}. Macro $\Delta$F1 is reported in percentage points after rejecting the most uncertain examples. "
            rf"\% Incorrect is the percentage of rejected examples that were originally misclassified. "
            rf"Values are mean and standard deviation across folds. "
            rf"\textbf{{Bold}} marks the best mean. "
            rf"\underline{{Underlined}} methods are statistically tied with the best method "
            rf"(Wilcoxon, $p\geq {alpha_tie:.2f}$) and within {macro_tolerance_pp:.2f} pp for Macro $\Delta$F1 "
            rf"or {pct_tolerance_pp:.2f} pp for \% Incorrect. "
            rf"$^{{\dagger}}$ indicates Wilcoxon $p<{significance_alpha:.2f}$ against zero for Macro $\Delta$F1 "
            rf"and against the fold baseline error rate for \% Incorrect."
        )

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")

    multiple_blocks = len(aggregates) > 1 or list(aggregates.keys()) != ["Overall"]

    for idx, (block_name, aggregate) in enumerate(aggregates.items()):
        title = str(block_name) if multiple_blocks else None
        write_rejection_block(
            lines,
            aggregate,
            methods=methods,
            rejection_rates=rejection_rates,
            block_title=title,
            significance_alpha=significance_alpha,
        )
        if idx < len(aggregates) - 1:
            lines.append(r"\vspace{0.4em}")

    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Dataset profile / IQR row
# ---------------------------------------------------------------------

def latex_iqr(q1: Optional[int], q3: Optional[int]) -> str:
    """Format an IQR range as q1--q3."""
    if q1 is None or q3 is None:
        return ""
    return f"{q1}--{q3}"


def build_iqr_latex_row(
    language_name: str,
    simple_n: int,
    simple_q1: Optional[int],
    simple_q3: Optional[int],
    complex_n: int,
    complex_q1: Optional[int],
    complex_q3: Optional[int],
) -> str:
    """Build a LaTeX row for sentence-length IQR dataset profiling."""
    return (
        f"{latex_escape(language_name)} & "
        f"{int(simple_n)} & {latex_iqr(simple_q1, simple_q3)} & "
        f"{int(complex_n)} & {latex_iqr(complex_q1, complex_q3)} \\\\"
    )


def build_dataset_heading_row(dataset_name: str, n_columns: int = 5) -> str:
    """Build a bold dataset heading row for a LaTeX table."""
    if n_columns < 1:
        raise ValueError("n_columns must be at least 1.")
    return rf"\textbf{{{latex_escape(dataset_name)}}}" + " & " * (n_columns - 1) + r" \\"


__all__ = [
    "DEFAULT_METHOD_ORDER",
    "METHOD_DISPLAY_NAMES",
    "METRIC_DISPLAY_NAMES",
    "METRIC_GROUPS",
    "latex_escape",
    "ordered_methods",
    "group_metrics",
    "display_method",
    "display_metric",
    "format_float",
    "style_text",
    "format_mean_with_style",
    "build_metric_summary_table",
    "build_metric_summary_table_from_wide_summary",
    "format_tau_with_p",
    "build_kendall_correlation_table",
    "build_pearson_correlation_table",
    "format_delta_mean",
    "format_delta_std",
    "format_pct_mean",
    "format_pct_std",
    "add_dagger_if_significant",
    "style_rejection_mean",
    "write_rejection_block",
    "build_rejection_table",
    "latex_iqr",
    "build_iqr_latex_row",
    "build_dataset_heading_row",
]
