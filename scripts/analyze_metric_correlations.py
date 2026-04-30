#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_metric_correlations.py

Compute correlations between uncertainty metrics across folds and methods.

This script takes fold-level metric summary files for one or more languages,
concatenates method values across folds, and computes:
  - Kendall's tau with p-values
  - Pearson's r

It can produce:
  - LaTeX tables for Kendall and Pearson correlations
  - per-language square correlation matrices as CSV
  - per-language Kendall tau heatmaps as PNG/PDF

Expected input format
---------------------
Each fold summary CSV should have:
  - rows = metrics
  - columns = uncertainty methods

Example
-------
python scripts/analyze_metric_correlations.py \
  --lang-glob "AR=results/fold_metrics/ar_fold*_metrics_summary*.csv" \
  --lang-glob "EN=results/fold_metrics/en_fold*_metrics_summary*.csv" \
  --lang-glob "FR=results/fold_metrics/fr_fold*_metrics_summary*.csv" \
  --lang-glob "HI=results/fold_metrics/hi_fold*_metrics_summary*.csv" \
  --lang-glob "RU=results/fold_metrics/ru_fold*_metrics_summary*.csv" \
  --outdir results/tables \
  --figdir results/figures/metric_correlations

For a single language:

python scripts/analyze_metric_correlations.py \
  --lang-glob "RU=results/fold_metrics/ru_fold*_metrics_summary*.csv" \
  --outdir results/tables
"""

from __future__ import annotations

import argparse
import glob
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr


# ---------------------------------------------------------------------
# Metric configuration
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
# Loading and metric-name helpers
# ---------------------------------------------------------------------

def first_present(canonical_metric: str, index_like: Iterable[str]) -> Optional[str]:
    """Find the first matching row name for a canonical metric using aliases."""
    aliases = METRIC_ALIASES.get(canonical_metric, [canonical_metric])
    lower_to_original = {str(x).lower(): str(x) for x in index_like}

    for alias in aliases:
        key = alias.lower()
        if key in lower_to_original:
            return lower_to_original[key]
    return None


def load_language_folds(glob_pattern: str) -> List[pd.DataFrame]:
    """Load all fold summary files for one language."""
    files = sorted(glob.glob(glob_pattern))
    if not files:
        return []

    fold_dfs: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce")

        # Drop duplicate metric rows, keeping the first occurrence.
        # This avoids issues such as both ROC-AUC and roc-auc appearing.
        df = df[~df.index.duplicated(keep="first")]
        fold_dfs.append(df)

    return fold_dfs


def load_all_languages(lang_globs: Dict[str, str]) -> Dict[str, List[pd.DataFrame]]:
    """Load fold summary files for all requested languages."""
    return {lang: load_language_folds(pattern) for lang, pattern in lang_globs.items()}


def common_methods_across_folds(fold_dfs: Sequence[pd.DataFrame]) -> List[str]:
    """Return methods present in all folds, preserving the first fold's order."""
    if not fold_dfs:
        return []

    first_methods = list(fold_dfs[0].columns)
    return [m for m in first_methods if all(m in df.columns for df in fold_dfs)]


def metric_present_in_all_folds(fold_dfs: Sequence[pd.DataFrame], canonical_metric: str) -> bool:
    """Return True if a canonical metric is present in every fold via aliases."""
    if not fold_dfs:
        return False

    for df in fold_dfs:
        row_name = first_present(canonical_metric, df.index)
        if row_name is None or row_name not in df.index:
            return False
    return True


# ---------------------------------------------------------------------
# Vector construction and correlations
# ---------------------------------------------------------------------

def concat_metric_vector_across_folds_and_methods(
    fold_dfs: Sequence[pd.DataFrame],
    canonical_metric: str,
) -> Optional[pd.Series]:
    """Build a long vector for one metric by concatenating method values across folds.

    Fold order is preserved. Within each fold, method order is the common method
    order from the first fold.
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

        s = df.loc[row_name, common_methods].copy()
        s.index = pd.MultiIndex.from_product([[fold_id], common_methods], names=["fold", "method"])
        vectors.append(s)

    return pd.concat(vectors, axis=0)


def compute_corr_for_pair_per_language(
    lang2folds: Dict[str, List[pd.DataFrame]],
    metric_1: str,
    metric_2: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
    """Compute Kendall tau, Kendall p, Pearson r, and n for one metric pair."""
    tau_values: Dict[str, float] = {}
    tau_p_values: Dict[str, float] = {}
    r_values: Dict[str, float] = {}
    n_values: Dict[str, int] = {}

    for lang, folds in lang2folds.items():
        v1 = concat_metric_vector_across_folds_and_methods(folds, metric_1)
        v2 = concat_metric_vector_across_folds_and_methods(folds, metric_2)

        if v1 is None or v2 is None:
            tau_values[lang] = np.nan
            tau_p_values[lang] = np.nan
            r_values[lang] = np.nan
            n_values[lang] = 0
            continue

        aligned = pd.concat([v1, v2], axis=1).dropna()
        aligned.columns = ["x", "y"]
        n = aligned.shape[0]
        n_values[lang] = int(n)

        if n < 2:
            tau_values[lang] = np.nan
            tau_p_values[lang] = np.nan
            r_values[lang] = np.nan
            continue

        x = aligned["x"].to_numpy(dtype=float)
        y = aligned["y"].to_numpy(dtype=float)

        try:
            tau, tau_p = kendalltau(x, y, nan_policy="omit")
            tau_values[lang] = float(tau)
            tau_p_values[lang] = float(tau_p) if np.isfinite(tau_p) else np.nan
        except Exception:
            tau_values[lang] = np.nan
            tau_p_values[lang] = np.nan

        try:
            r, _ = pearsonr(x, y)
            r_values[lang] = float(r)
        except Exception:
            r_values[lang] = np.nan

    return tau_values, tau_p_values, r_values, n_values


def discover_available_metrics(
    lang2folds: Dict[str, List[pd.DataFrame]],
    candidate_metrics: Sequence[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Find canonical metrics available in every fold for each language."""
    per_lang_available: Dict[str, List[str]] = {}
    union_available = set()

    for lang, folds in lang2folds.items():
        if not folds:
            per_lang_available[lang] = []
            continue

        common_methods = common_methods_across_folds(folds)
        if len(common_methods) < 2:
            per_lang_available[lang] = []
            continue

        available = [m for m in candidate_metrics if metric_present_in_all_folds(folds, m)]
        per_lang_available[lang] = sorted(available)
        union_available.update(available)

    return per_lang_available, sorted(union_available)


# ---------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------

def build_correlation_tables(
    lang2folds: Dict[str, List[pd.DataFrame]],
    use_all_metrics: bool = True,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, List[str]],
]:
    """Build Kendall, p-value, Pearson, and n tables.

    Returns:
      tau_tables, tau_p_tables, r_tables, n_tables, per_lang_available
    """
    langs = list(lang2folds.keys())
    candidates = list(METRIC_ALIASES.keys())
    per_lang_available, union_available = discover_available_metrics(lang2folds, candidates)

    if use_all_metrics:
        metric_pairs_by_group = {
            "All Metrics": list(combinations(union_available, 2)),
        }
    else:
        metric_pairs_by_group = CURATED_METRIC_PAIRS_BY_GROUP

    tau_tables: Dict[str, pd.DataFrame] = {}
    tau_p_tables: Dict[str, pd.DataFrame] = {}
    r_tables: Dict[str, pd.DataFrame] = {}
    n_tables: Dict[str, pd.DataFrame] = {}

    for group, pairs in metric_pairs_by_group.items():
        idx = pd.MultiIndex.from_tuples(pairs, names=["metric_1", "metric_2"])
        tau_tables[group] = pd.DataFrame(index=idx, columns=langs, dtype=float)
        tau_p_tables[group] = pd.DataFrame(index=idx, columns=langs, dtype=float)
        r_tables[group] = pd.DataFrame(index=idx, columns=langs, dtype=float)
        n_tables[group] = pd.DataFrame(index=idx, columns=langs, dtype=float)

        for metric_1, metric_2 in pairs:
            tau, tau_p, r, n = compute_corr_for_pair_per_language(lang2folds, metric_1, metric_2)
            for lang in langs:
                tau_tables[group].loc[(metric_1, metric_2), lang] = tau.get(lang, np.nan)
                tau_p_tables[group].loc[(metric_1, metric_2), lang] = tau_p.get(lang, np.nan)
                r_tables[group].loc[(metric_1, metric_2), lang] = r.get(lang, np.nan)
                n_tables[group].loc[(metric_1, metric_2), lang] = n.get(lang, 0)

    return tau_tables, tau_p_tables, r_tables, n_tables, per_lang_available


def format_tau_with_p(tau: float, p_value: float) -> str:
    if pd.isna(tau):
        return "--"

    s = f"{tau:.2f}"
    if np.isfinite(p_value):
        if p_value < 0.01:
            return rf"\textbf{{{s}}}"
        if p_value < 0.05:
            return rf"\underline{{{s}}}"
    return s


def format_r(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.2f}"


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


def build_tau_latex_table(
    tau_tables: Dict[str, pd.DataFrame],
    tau_p_tables: Dict[str, pd.DataFrame],
    caption: str,
    label: str,
) -> str:
    """Build LaTeX table for Kendall tau values."""
    first_group = next(iter(tau_tables))
    langs = list(tau_tables[first_group].columns)

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
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
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def build_r_latex_table(
    r_tables: Dict[str, pd.DataFrame],
    caption: str,
    label: str,
) -> str:
    """Build LaTeX table for Pearson r values."""
    first_group = next(iter(r_tables))
    langs = list(r_tables[first_group].columns)

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
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
                row.append(format_r(table.loc[(metric_1, metric_2), lang]))
            lines.append(" & ".join(row) + r" \\")

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Matrix and plotting helpers
# ---------------------------------------------------------------------

def correlation_matrix_from_pair_table(
    pair_table: pd.DataFrame,
    metrics: Sequence[str],
    lang: str,
) -> Optional[pd.DataFrame]:
    """Convert a pairwise correlation table into a square matrix for one language."""
    if len(metrics) < 2 or lang not in pair_table.columns:
        return None

    mat = pd.DataFrame(index=list(metrics), columns=list(metrics), dtype=float)

    for metric in metrics:
        mat.loc[metric, metric] = 1.0

    for metric_1, metric_2 in combinations(metrics, 2):
        if (metric_1, metric_2) in pair_table.index:
            value = pair_table.loc[(metric_1, metric_2), lang]
        elif (metric_2, metric_1) in pair_table.index:
            value = pair_table.loc[(metric_2, metric_1), lang]
        else:
            value = np.nan

        mat.loc[metric_1, metric_2] = value
        mat.loc[metric_2, metric_1] = value

    return mat


def save_heatmap(
    matrix: pd.DataFrame,
    title: str,
    outpath: Path,
    dpi: int = 300,
    annotate: bool = True,
) -> None:
    """Save a square heatmap for a correlation matrix."""
    metrics = list(matrix.index)
    matrix = matrix.loc[metrics, metrics]

    size = max(6.0, 0.55 * len(metrics))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(matrix.values, interpolation="nearest", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(metrics)

    if annotate and len(metrics) <= 12:
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="both", length=0)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_tables_and_matrices(
    tau_tables: Dict[str, pd.DataFrame],
    tau_p_tables: Dict[str, pd.DataFrame],
    r_tables: Dict[str, pd.DataFrame],
    n_tables: Dict[str, pd.DataFrame],
    per_lang_available: Dict[str, List[str]],
    outdir: Path,
    figdir: Path,
    save_plots: bool,
    dpi: int,
) -> None:
    """Write LaTeX, CSV tables, matrices, and optional plots."""
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    tau_caption = (
        r"Kendall's correlation ($\tau$) between uncertainty metric pairs, computed by "
        r"concatenating method-level values across folds for each language. "
        r"Bold indicates $p<0.01$; underline indicates $p<0.05$."
    )
    r_caption = (
        r"Pearson correlation ($r$) between uncertainty metric pairs, computed by "
        r"concatenating method-level values across folds for each language."
    )

    tau_latex = build_tau_latex_table(
        tau_tables,
        tau_p_tables,
        caption=tau_caption,
        label="tab:metric_correlations_kendall",
    )
    r_latex = build_r_latex_table(
        r_tables,
        caption=r_caption,
        label="tab:metric_correlations_pearson",
    )

    (outdir / "metric_correlations_kendall.tex").write_text(tau_latex, encoding="utf-8")
    (outdir / "metric_correlations_pearson.tex").write_text(r_latex, encoding="utf-8")

    for group, table in tau_tables.items():
        slug = group.lower().replace(" ", "_")
        table.to_csv(outdir / f"kendall_tau_pairs_{slug}.csv")
        tau_p_tables[group].to_csv(outdir / f"kendall_tau_pvalues_{slug}.csv")
        r_tables[group].to_csv(outdir / f"pearson_r_pairs_{slug}.csv")
        n_tables[group].to_csv(outdir / f"correlation_n_pairs_{slug}.csv")

    # For matrices and heatmaps, use the first available group. In all-metrics mode,
    # this is 'All Metrics'. In curated mode, matrices may be sparse because only
    # selected pairs are available.
    first_group = next(iter(tau_tables))
    tau_pair_table = tau_tables[first_group]
    r_pair_table = r_tables[first_group]

    for lang, metrics in per_lang_available.items():
        if len(metrics) < 2:
            continue

        tau_mat = correlation_matrix_from_pair_table(tau_pair_table, metrics, lang)
        r_mat = correlation_matrix_from_pair_table(r_pair_table, metrics, lang)

        if tau_mat is not None:
            tau_mat.to_csv(outdir / f"tau_matrix_{lang}.csv")
            if save_plots:
                save_heatmap(
                    tau_mat,
                    title=rf"Kendall's $\tau$ — {lang}",
                    outpath=figdir / f"tau_heatmap_{lang}.png",
                    dpi=dpi,
                    annotate=True,
                )

        if r_mat is not None:
            r_mat.to_csv(outdir / f"pearson_r_matrix_{lang}.csv")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_lang_globs(items: Sequence[str]) -> Dict[str, str]:
    """Parse repeated LANG=GLOB arguments."""
    parsed: Dict[str, str] = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --lang-glob value: {item!r}. Expected LANG=GLOB.")
        lang, pattern = item.split("=", 1)
        lang = lang.strip()
        pattern = pattern.strip()
        if not lang or not pattern:
            raise ValueError(f"Invalid --lang-glob value: {item!r}. Expected LANG=GLOB.")
        parsed[lang] = pattern

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute correlations between uncertainty metrics across folds and methods."
    )
    parser.add_argument(
        "--lang-glob",
        action="append",
        required=True,
        help=(
            "Language-specific glob in the form LANG=GLOB. "
            "Repeat for multiple languages, e.g. --lang-glob 'RU=results/fold_metrics/ru_fold*.csv'."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="results/tables",
        help="Output directory for LaTeX and CSV files.",
    )
    parser.add_argument(
        "--figdir",
        default=None,
        help="Output directory for heatmaps. Default: <outdir>/figs.",
    )
    parser.add_argument(
        "--curated-only",
        action="store_true",
        help="Use only the curated metric pairs instead of all discoverable metric pairs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save heatmap figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved heatmaps. Default: 300.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lang_globs = parse_lang_globs(args.lang_glob)

    outdir = Path(args.outdir)
    figdir = Path(args.figdir) if args.figdir else outdir / "figs"

    lang2folds = load_all_languages(lang_globs)

    for lang, folds in lang2folds.items():
        print(f"[info] {lang}: loaded {len(folds)} fold file(s).")
        if not folds:
            print(f"[warn] No files found for {lang}: {lang_globs[lang]}")

    tau_tables, tau_p_tables, r_tables, n_tables, per_lang_available = build_correlation_tables(
        lang2folds,
        use_all_metrics=not args.curated_only,
    )

    save_tables_and_matrices(
        tau_tables=tau_tables,
        tau_p_tables=tau_p_tables,
        r_tables=r_tables,
        n_tables=n_tables,
        per_lang_available=per_lang_available,
        outdir=outdir,
        figdir=figdir,
        save_plots=not args.no_plots,
        dpi=args.dpi,
    )

    print(f"[ok] Wrote correlation tables to: {outdir}")
    if not args.no_plots:
        print(f"[ok] Wrote heatmaps to: {figdir}")

    print("[info] Available metrics per language:")
    for lang, metrics in per_lang_available.items():
        print(f"       {lang}: {', '.join(metrics) if metrics else 'none'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
