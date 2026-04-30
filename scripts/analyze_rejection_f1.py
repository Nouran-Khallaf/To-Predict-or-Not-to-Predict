#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_rejection_f1.py

Evaluate uncertainty-guided rejection / selective prediction.

For each fold and each uncertainty method, the script rejects the most uncertain
examples at fixed rejection rates, then reports:

  1. Macro-F1 change after rejection: Macro ΔF1 = F1(kept) - F1(original)
  2. Percentage of incorrect predictions among the rejected examples
  3. Rejected counts and incorrect-rejected counts

The LaTeX table uses:
  - bold = best mean at a given rejection rate
  - underline = statistically tied with the best method and practically close
  - dagger = significantly better than the baseline comparison

Expected input format
---------------------
Each fold prediction CSV should contain:
  - true label column, default: true_label
  - predicted label column, default: predicted_label
  - one column per uncertainty method, e.g. SR, SMP, ENT, ENT_MC, PV, BALD, MD, HUQ-MD, LOF, ISOF

By default, larger uncertainty score = more uncertain.
For LOF and ISOF, the score is multiplied by -1 unless disabled, because these
methods often use the opposite direction.

Example
-------
python scripts/analyze_rejection_f1.py \
  --input-glob "results/predictions/ru-uncertainty_results_fold*.csv" \
  --lang RU \
  --outdir results/tables

With language blocks inside each prediction file:

python scripts/analyze_rejection_f1.py \
  --input-glob "results/predictions/all-uncertainty_results_fold*.csv" \
  --lang ALL \
  --block-by-language \
  --outdir results/tables
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_METHODS = [
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
    "SR": "SR",
    "SMP": "SMP",
    "ENT": "ENT",
    "PV": "PV",
    "BALD": "BALD",
    "MD": "MD",
    "HUQ-MD": "HUQ-MD",
    "LOF": "LOF",
    "ISOF": "ISOF",
}

DEFAULT_REJECTION_RATES = [0.01, 0.05, 0.10, 0.15]
DEFAULT_REVERSE_SCORE_METHODS = {"LOF", "ISOF"}


# ---------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------

def safe_wilcoxon_against_const(values: Sequence[float], baseline: float | Sequence[float]) -> float:
    """Wilcoxon signed-rank test of values against a constant or paired baseline."""
    v = np.asarray(values, dtype=float)

    if np.ndim(baseline) == 0:
        b = np.full_like(v, float(baseline))
    else:
        b = np.asarray(baseline, dtype=float)

    mask = np.isfinite(v) & np.isfinite(b)
    v = v[mask]
    b = b[mask]

    if len(v) < 2:
        return np.nan

    diffs = v - b
    if np.allclose(diffs, 0.0, atol=0.0, rtol=0.0):
        return 1.0

    try:
        _, p_value = wilcoxon(
            diffs,
            zero_method="zsplit",
            alternative="two-sided",
            correction=False,
            mode="auto",
        )
        return float(p_value)
    except Exception:
        return np.nan


def safe_wilcoxon_pair(x: Sequence[float], y: Sequence[float]) -> float:
    """Paired Wilcoxon signed-rank test between two vectors."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    if len(x_arr) < 2:
        return np.nan

    diffs = x_arr - y_arr
    if np.allclose(diffs, 0.0, atol=0.0, rtol=0.0):
        return 1.0

    try:
        _, p_value = wilcoxon(
            diffs,
            zero_method="zsplit",
            alternative="two-sided",
            correction=False,
            mode="auto",
        )
        return float(p_value)
    except Exception:
        return np.nan


def pairwise_tie_mask_vs_best(
    data_tensor: np.ndarray,
    mean_matrix: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find best method and methods statistically tied with best.

    Parameters
    ----------
    data_tensor:
        Shape = methods x folds x rejection_rates
    mean_matrix:
        Shape = methods x rejection_rates
    alpha:
        Significance level. A method is tied with best when p >= alpha.
    """
    n_methods, _, n_rates = data_tensor.shape
    best_idx = np.full(n_rates, -1, dtype=int)
    tied_mask = np.zeros((n_methods, n_rates), dtype=bool)

    for rate_idx in range(n_rates):
        means = mean_matrix[:, rate_idx]
        if not np.any(np.isfinite(means)):
            continue

        best = int(np.nanargmax(means))
        best_idx[rate_idx] = best
        best_vec = data_tensor[best, :, rate_idx]

        for method_idx in range(n_methods):
            if method_idx == best:
                continue

            vec = data_tensor[method_idx, :, rate_idx]
            p_value = safe_wilcoxon_pair(vec, best_vec)
            if np.isfinite(p_value) and p_value >= alpha:
                tied_mask[method_idx, rate_idx] = True

    return best_idx, tied_mask


def refine_ties_with_tolerance(
    mean_matrix: np.ndarray,
    best_idx: np.ndarray,
    tied_mask: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Keep only statistical ties that are also practically close to the best mean."""
    refined = np.zeros_like(tied_mask, dtype=bool)
    n_methods, n_rates = mean_matrix.shape

    for rate_idx in range(n_rates):
        best = best_idx[rate_idx]
        if best < 0 or not np.isfinite(mean_matrix[best, rate_idx]):
            continue

        best_mean = mean_matrix[best, rate_idx]
        diffs = np.abs(mean_matrix[:, rate_idx] - best_mean)
        keep = tied_mask[:, rate_idx] & np.isfinite(diffs) & (diffs <= tolerance)
        refined[:, rate_idx] = keep

    return refined


# ---------------------------------------------------------------------
# Core rejection computation
# ---------------------------------------------------------------------

def validate_prediction_frame(
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    path: str | Path,
) -> None:
    required = {true_col, pred_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s) {sorted(missing)} in {path}")


def compute_rejection_for_frame(
    df: pd.DataFrame,
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    true_col: str,
    pred_col: str,
    reverse_score_methods: Iterable[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], float]:
    """Compute rejection statistics for one fold/frame.

    Returns four method dictionaries, each containing arrays over rejection rates:
      - macro_f1_delta
      - rejected_error_rate
      - rejected_count
      - incorrect_rejected_count
    plus the fold's baseline error percentage.
    """
    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy()

    original_macro_f1 = f1_score(y_true, y_pred, average="macro")
    baseline_error_pct = 100.0 * np.mean(y_pred != y_true)

    reverse_score_methods = set(reverse_score_methods)

    macro_f1_delta: Dict[str, np.ndarray] = {}
    rejected_error_rate: Dict[str, np.ndarray] = {}
    rejected_count: Dict[str, np.ndarray] = {}
    incorrect_rejected_count: Dict[str, np.ndarray] = {}

    n_examples = len(df)

    for method in methods:
        if method not in df.columns:
            n_rates = len(rejection_rates)
            macro_f1_delta[method] = np.full(n_rates, np.nan)
            rejected_error_rate[method] = np.full(n_rates, np.nan)
            rejected_count[method] = np.full(n_rates, np.nan)
            incorrect_rejected_count[method] = np.full(n_rates, np.nan)
            continue

        scores = df[method].to_numpy(dtype=float)
        if method in reverse_score_methods:
            scores = -scores

        # Small scores first, largest scores last. Largest = most uncertain.
        order = np.argsort(scores)

        deltas: List[float] = []
        pct_bad: List[float] = []
        counts: List[int] = []
        incorrect_counts: List[int] = []

        for rate in rejection_rates:
            n_reject = int(math.ceil(rate * n_examples))
            n_reject = max(0, min(n_reject, n_examples))

            if n_reject == 0:
                deltas.append(0.0)
                pct_bad.append(np.nan)
                counts.append(0)
                incorrect_counts.append(0)
                continue

            rejected_idx = order[-n_reject:]
            kept_idx = order[:-n_reject]

            if len(kept_idx) == 0:
                kept_macro_f1 = np.nan
            else:
                kept_macro_f1 = f1_score(y_true[kept_idx], y_pred[kept_idx], average="macro")

            deltas.append(kept_macro_f1 - original_macro_f1)

            incorrect = int(np.sum(y_pred[rejected_idx] != y_true[rejected_idx]))
            pct_bad.append(100.0 * incorrect / n_reject)
            counts.append(n_reject)
            incorrect_counts.append(incorrect)

        macro_f1_delta[method] = np.asarray(deltas, dtype=float)
        rejected_error_rate[method] = np.asarray(pct_bad, dtype=float)
        rejected_count[method] = np.asarray(counts, dtype=float)
        incorrect_rejected_count[method] = np.asarray(incorrect_counts, dtype=float)

    return (
        macro_f1_delta,
        rejected_error_rate,
        rejected_count,
        incorrect_rejected_count,
        baseline_error_pct,
    )


def available_methods(files: Sequence[Path], requested_methods: Sequence[str]) -> List[str]:
    """Return requested methods that appear in at least one input file."""
    found = set()
    for path in files:
        try:
            cols = pd.read_csv(path, nrows=0).columns
            found.update(m for m in requested_methods if m in cols)
        except Exception:
            continue

    ordered = [m for m in requested_methods if m in found]
    return ordered if ordered else list(requested_methods)


def collect_overall_data(
    files: Sequence[Path],
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    true_col: str,
    pred_col: str,
    reverse_score_methods: Iterable[str],
) -> Dict[str, object]:
    """Collect rejection statistics across folds."""
    macro_f1_delta = {m: [] for m in methods}
    rejected_error_rate = {m: [] for m in methods}
    rejected_count = {m: [] for m in methods}
    incorrect_rejected_count = {m: [] for m in methods}
    baseline_error_pct_per_fold: List[float] = []

    for path in files:
        df = pd.read_csv(path)
        validate_prediction_frame(df, true_col, pred_col, path)

        d_macro, pct_bad, n_rej, n_bad, base_err = compute_rejection_for_frame(
            df=df,
            methods=methods,
            rejection_rates=rejection_rates,
            true_col=true_col,
            pred_col=pred_col,
            reverse_score_methods=reverse_score_methods,
        )

        for method in methods:
            macro_f1_delta[method].append(d_macro[method])
            rejected_error_rate[method].append(pct_bad[method])
            rejected_count[method].append(n_rej[method])
            incorrect_rejected_count[method].append(n_bad[method])

        baseline_error_pct_per_fold.append(base_err)

    return {
        "macro_f1_delta": macro_f1_delta,
        "rejected_error_rate": rejected_error_rate,
        "rejected_count": rejected_count,
        "incorrect_rejected_count": incorrect_rejected_count,
        "baseline_error_pct_per_fold": np.asarray(baseline_error_pct_per_fold, dtype=float),
    }


def collect_language_block_data(
    files: Sequence[Path],
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    true_col: str,
    pred_col: str,
    language_col: str,
    reverse_score_methods: Iterable[str],
) -> Dict[str, Dict[str, object]]:
    """Collect rejection statistics grouped by language column."""
    per_language: Dict[str, Dict[str, object]] = {}

    def ensure_language(lang: str) -> Dict[str, object]:
        if lang not in per_language:
            per_language[lang] = {
                "macro_f1_delta": {m: [] for m in methods},
                "rejected_error_rate": {m: [] for m in methods},
                "rejected_count": {m: [] for m in methods},
                "incorrect_rejected_count": {m: [] for m in methods},
                "baseline_error_pct_per_fold": [],
            }
        return per_language[lang]

    for path in files:
        df = pd.read_csv(path)
        validate_prediction_frame(df, true_col, pred_col, path)

        if language_col not in df.columns:
            raise ValueError(f"--block-by-language was set, but column {language_col!r} is missing in {path}")

        for lang_value, df_lang in df.groupby(language_col):
            lang_key = str(lang_value)
            store = ensure_language(lang_key)

            d_macro, pct_bad, n_rej, n_bad, base_err = compute_rejection_for_frame(
                df=df_lang,
                methods=methods,
                rejection_rates=rejection_rates,
                true_col=true_col,
                pred_col=pred_col,
                reverse_score_methods=reverse_score_methods,
            )

            for method in methods:
                store["macro_f1_delta"][method].append(d_macro[method])
                store["rejected_error_rate"][method].append(pct_bad[method])
                store["rejected_count"][method].append(n_rej[method])
                store["incorrect_rejected_count"][method].append(n_bad[method])

            store["baseline_error_pct_per_fold"].append(base_err)

    for lang_store in per_language.values():
        lang_store["baseline_error_pct_per_fold"] = np.asarray(
            lang_store["baseline_error_pct_per_fold"], dtype=float
        )

    return per_language


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def stack_metric_dict(data_dict: Dict[str, List[np.ndarray]], methods: Sequence[str], n_rates: int) -> np.ndarray:
    """Stack method -> list(fold arrays) into method x fold x rejection-rate tensor."""
    tensors: List[np.ndarray] = []

    for method in methods:
        arrays = data_dict.get(method, [])
        if arrays:
            tensors.append(np.vstack(arrays))
        else:
            tensors.append(np.full((0, n_rates), np.nan))

    return np.stack(tensors, axis=0)


def aggregate_block(
    block_data: Dict[str, object],
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    alpha_tie: float,
    macro_tolerance_pp: float,
    pct_tolerance_pp: float,
) -> Dict[str, object]:
    """Aggregate one overall or per-language block."""
    n_rates = len(rejection_rates)

    a_macro_raw = stack_metric_dict(block_data["macro_f1_delta"], methods, n_rates)
    a_macro = a_macro_raw * 100.0  # percentage points
    a_pct = stack_metric_dict(block_data["rejected_error_rate"], methods, n_rates)
    a_rej = stack_metric_dict(block_data["rejected_count"], methods, n_rates)
    a_inc = stack_metric_dict(block_data["incorrect_rejected_count"], methods, n_rates)
    baseline_error_pct = np.asarray(block_data["baseline_error_pct_per_fold"], dtype=float)

    mean_macro = np.nanmean(a_macro, axis=1)
    std_macro = np.nanstd(a_macro, axis=1)
    mean_pct = np.nanmean(a_pct, axis=1)
    std_pct = np.nanstd(a_pct, axis=1)

    p_macro = np.full_like(mean_macro, np.nan, dtype=float)
    p_pct = np.full_like(mean_pct, np.nan, dtype=float)

    for method_idx in range(len(methods)):
        for rate_idx in range(n_rates):
            p_macro[method_idx, rate_idx] = safe_wilcoxon_against_const(
                a_macro[method_idx, :, rate_idx],
                0.0,
            )
            p_pct[method_idx, rate_idx] = safe_wilcoxon_against_const(
                a_pct[method_idx, :, rate_idx],
                baseline_error_pct,
            )

    best_macro_idx, tied_macro_mask = pairwise_tie_mask_vs_best(a_macro, mean_macro, alpha=alpha_tie)
    best_pct_idx, tied_pct_mask = pairwise_tie_mask_vs_best(a_pct, mean_pct, alpha=alpha_tie)

    tied_macro_mask = refine_ties_with_tolerance(
        mean_macro,
        best_macro_idx,
        tied_macro_mask,
        tolerance=macro_tolerance_pp,
    )
    tied_pct_mask = refine_ties_with_tolerance(
        mean_pct,
        best_pct_idx,
        tied_pct_mask,
        tolerance=pct_tolerance_pp,
    )

    return {
        "a_macro": a_macro,
        "a_pct": a_pct,
        "a_rej": a_rej,
        "a_inc": a_inc,
        "baseline_error_pct_per_fold": baseline_error_pct,
        "mean_macro": mean_macro,
        "std_macro": std_macro,
        "mean_pct": mean_pct,
        "std_pct": std_pct,
        "p_macro": p_macro,
        "p_pct": p_pct,
        "best_macro_idx": best_macro_idx,
        "tied_macro_mask": tied_macro_mask,
        "best_pct_idx": best_pct_idx,
        "tied_pct_mask": tied_pct_mask,
    }


def build_summary_rows(
    aggregate: Dict[str, object],
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    block_name: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Build tidy rows for CSV output."""
    rows: List[Dict[str, object]] = []

    mean_macro = aggregate["mean_macro"]
    std_macro = aggregate["std_macro"]
    mean_pct = aggregate["mean_pct"]
    std_pct = aggregate["std_pct"]
    p_macro = aggregate["p_macro"]
    p_pct = aggregate["p_pct"]
    a_rej = aggregate["a_rej"]
    a_inc = aggregate["a_inc"]

    for method_idx, method in enumerate(methods):
        for rate_idx, rate in enumerate(rejection_rates):
            row = {
                "method": method,
                "rejection_rate": rate,
                "macro_delta_mean_pp": mean_macro[method_idx, rate_idx],
                "macro_delta_std_pp": std_macro[method_idx, rate_idx],
                "macro_delta_pvalue_vs_zero": p_macro[method_idx, rate_idx],
                "pct_incorrect_rejected_mean": mean_pct[method_idx, rate_idx],
                "pct_incorrect_rejected_std": std_pct[method_idx, rate_idx],
                "pct_incorrect_vs_baseline_pvalue": p_pct[method_idx, rate_idx],
                "rejected_mean_count": np.nanmean(a_rej[method_idx, :, rate_idx]),
                "rejected_std_count": np.nanstd(a_rej[method_idx, :, rate_idx]),
                "incorrect_rejected_mean_count": np.nanmean(a_inc[method_idx, :, rate_idx]),
                "incorrect_rejected_std_count": np.nanstd(a_inc[method_idx, :, rate_idx]),
            }
            if block_name is not None:
                row["block"] = block_name
            rows.append(row)

    return rows


# ---------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------

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


def format_delta_mean(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) < 0.005:
        return "0"
    return f"{value:.2f}"


def format_delta_std(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def format_pct_mean(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.1f}"


def format_pct_std(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.1f}"


def style_mean(
    text: str,
    is_best: bool,
    is_tied: bool,
    p_value: float,
) -> str:
    out = text

    if np.isfinite(p_value) and p_value < 0.05:
        out += r"$^{\dagger}$"

    if is_best:
        return rf"\textbf{{{out}}}"
    if is_tied:
        return rf"\underline{{{out}}}"
    return out


def write_latex_block(
    lines: List[str],
    aggregate: Dict[str, object],
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    block_title: Optional[str] = None,
) -> None:
    """Append one LaTeX block/table body."""
    n_rates = len(rejection_rates)
    col_format = "l" + " " + " ".join(
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

    mean_macro = aggregate["mean_macro"]
    std_macro = aggregate["std_macro"]
    mean_pct = aggregate["mean_pct"]
    std_pct = aggregate["std_pct"]
    p_macro = aggregate["p_macro"]
    p_pct = aggregate["p_pct"]
    best_macro_idx = aggregate["best_macro_idx"]
    tied_macro_mask = aggregate["tied_macro_mask"]
    best_pct_idx = aggregate["best_pct_idx"]
    tied_pct_mask = aggregate["tied_pct_mask"]

    for method_idx, method in enumerate(methods):
        display_name = METHOD_DISPLAY_NAMES.get(method, method)
        cells = [latex_escape(display_name)]

        for rate_idx in range(n_rates):
            macro_mu = format_delta_mean(mean_macro[method_idx, rate_idx])
            macro_sd = format_delta_std(std_macro[method_idx, rate_idx])
            macro_mu = style_mean(
                macro_mu,
                is_best=(method_idx == best_macro_idx[rate_idx]),
                is_tied=(method_idx != best_macro_idx[rate_idx] and tied_macro_mask[method_idx, rate_idx]),
                p_value=p_macro[method_idx, rate_idx],
            )
            cells.extend([macro_mu, macro_sd])

            pct_mu = format_pct_mean(mean_pct[method_idx, rate_idx])
            pct_sd = format_pct_std(std_pct[method_idx, rate_idx])
            pct_mu = style_mean(
                pct_mu,
                is_best=(method_idx == best_pct_idx[rate_idx]),
                is_tied=(method_idx != best_pct_idx[rate_idx] and tied_pct_mask[method_idx, rate_idx]),
                p_value=p_pct[method_idx, rate_idx],
            )
            cells.extend([pct_mu, pct_sd])

        lines.append(" " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")


def build_latex_table(
    aggregates: Dict[str, Dict[str, object]],
    methods: Sequence[str],
    rejection_rates: Sequence[float],
    lang: str,
    alpha_tie: float,
    macro_tolerance_pp: float,
    pct_tolerance_pp: float,
    label: Optional[str] = None,
) -> str:
    """Build full LaTeX table."""
    if label is None:
        label = f"tab:rejection_f1_{lang.lower().replace(' ', '_')}"

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")

    multiple_blocks = len(aggregates) > 1 or list(aggregates.keys()) != ["Overall"]

    for idx, (block_name, aggregate) in enumerate(aggregates.items()):
        title = block_name if multiple_blocks else None
        write_latex_block(lines, aggregate, methods, rejection_rates, block_title=title)
        if idx < len(aggregates) - 1:
            lines.append(r"\vspace{0.4em}")

    lines.append(
        rf"\caption{{{latex_escape(lang)} uncertainty-guided rejection results. "
        rf"Macro $\Delta$F1 is reported in percentage points after rejecting the most uncertain examples. "
        rf"\% Incorrect is the percentage of rejected examples that were originally misclassified. "
        rf"Values are mean and standard deviation across folds. "
        rf"\textbf{{Bold}} marks the best mean. "
        rf"\underline{{Underlined}} methods are statistically tied with the best method "
        rf"(Wilcoxon, $p\geq {alpha_tie:.2f}$) and within {macro_tolerance_pp:.2f} pp for Macro $\Delta$F1 "
        rf"or {pct_tolerance_pp:.2f} pp for \% Incorrect. "
        rf"$^{{\dagger}}$ indicates Wilcoxon $p<0.05$ against zero for Macro $\Delta$F1 "
        rf"and against the fold baseline error rate for \% Incorrect.}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_methods(methods_text: Optional[str]) -> List[str]:
    if not methods_text:
        return list(DEFAULT_METHODS)
    return [x.strip() for x in methods_text.split(",") if x.strip()]


def parse_rates(rates_text: Optional[str]) -> List[float]:
    if not rates_text:
        return list(DEFAULT_REJECTION_RATES)
    rates = [float(x.strip()) for x in rates_text.split(",") if x.strip()]
    for rate in rates:
        if rate < 0 or rate > 1:
            raise ValueError(f"Invalid rejection rate {rate}. Rates must be between 0 and 1.")
    return rates


def parse_reverse_methods(text: Optional[str]) -> List[str]:
    if text is None:
        return sorted(DEFAULT_REVERSE_SCORE_METHODS)
    if text.strip() == "":
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty-guided rejection using macro-F1 and rejected-error rates."
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob pattern for fold-level prediction CSV files.",
    )
    parser.add_argument(
        "--lang",
        required=True,
        help="Language code/name used in output filenames and captions, e.g. RU.",
    )
    parser.add_argument(
        "--outdir",
        default="results/tables",
        help="Output directory for LaTeX and CSV files.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated uncertainty method columns. Default uses common benchmark methods.",
    )
    parser.add_argument(
        "--rejection-rates",
        default=None,
        help="Comma-separated rejection rates. Default: 0.01,0.05,0.10,0.15.",
    )
    parser.add_argument(
        "--true-col",
        default="true_label",
        help="Ground-truth label column. Default: true_label.",
    )
    parser.add_argument(
        "--pred-col",
        default="predicted_label",
        help="Predicted label column. Default: predicted_label.",
    )
    parser.add_argument(
        "--reverse-score-methods",
        default=None,
        help=(
            "Comma-separated methods whose scores should be multiplied by -1. "
            "Default: LOF,ISOF. Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--block-by-language",
        action="store_true",
        help="Create separate table blocks using a language column in the prediction files.",
    )
    parser.add_argument(
        "--language-col",
        default="language",
        help="Language column used with --block-by-language. Default: language.",
    )
    parser.add_argument(
        "--alpha-tie",
        type=float,
        default=0.05,
        help="Alpha for identifying statistical ties with the best method. Default: 0.05.",
    )
    parser.add_argument(
        "--macro-tolerance-pp",
        type=float,
        default=0.10,
        help="Practical tolerance for underlining Macro ΔF1 ties, in percentage points. Default: 0.10.",
    )
    parser.add_argument(
        "--pct-tolerance-pp",
        type=float,
        default=1.00,
        help="Practical tolerance for underlining %% Incorrect ties, in percentage points. Default: 1.00.",
    )
    parser.add_argument(
        "--latex-label",
        default=None,
        help="Optional custom LaTeX label.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    files = sorted(Path(p) for p in glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"No prediction files matched: {args.input_glob}")

    requested_methods = parse_methods(args.methods)
    methods = available_methods(files, requested_methods)
    rejection_rates = parse_rates(args.rejection_rates)
    reverse_score_methods = parse_reverse_methods(args.reverse_score_methods)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Found {len(files)} prediction file(s).")
    for p in files:
        print(f"       - {p}")
    print(f"[info] Methods: {', '.join(methods)}")
    print(f"[info] Rejection rates: {', '.join(str(r) for r in rejection_rates)}")
    print(f"[info] Reversed-score methods: {', '.join(reverse_score_methods) if reverse_score_methods else 'none'}")

    lang_slug = args.lang.lower().replace(" ", "_")

    if args.block_by_language:
        per_lang_data = collect_language_block_data(
            files=files,
            methods=methods,
            rejection_rates=rejection_rates,
            true_col=args.true_col,
            pred_col=args.pred_col,
            language_col=args.language_col,
            reverse_score_methods=reverse_score_methods,
        )

        aggregates: Dict[str, Dict[str, object]] = {}
        all_rows: List[Dict[str, object]] = []
        for block_name, block_data in sorted(per_lang_data.items()):
            aggregate = aggregate_block(
                block_data,
                methods=methods,
                rejection_rates=rejection_rates,
                alpha_tie=args.alpha_tie,
                macro_tolerance_pp=args.macro_tolerance_pp,
                pct_tolerance_pp=args.pct_tolerance_pp,
            )
            aggregates[block_name] = aggregate
            all_rows.extend(build_summary_rows(aggregate, methods, rejection_rates, block_name=block_name))
    else:
        overall_data = collect_overall_data(
            files=files,
            methods=methods,
            rejection_rates=rejection_rates,
            true_col=args.true_col,
            pred_col=args.pred_col,
            reverse_score_methods=reverse_score_methods,
        )
        aggregate = aggregate_block(
            overall_data,
            methods=methods,
            rejection_rates=rejection_rates,
            alpha_tie=args.alpha_tie,
            macro_tolerance_pp=args.macro_tolerance_pp,
            pct_tolerance_pp=args.pct_tolerance_pp,
        )
        aggregates = {"Overall": aggregate}
        all_rows = build_summary_rows(aggregate, methods, rejection_rates, block_name=None)

    summary_df = pd.DataFrame(all_rows)
    summary_csv = outdir / f"{lang_slug}_rejection_counts_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    latex_table = build_latex_table(
        aggregates=aggregates,
        methods=methods,
        rejection_rates=rejection_rates,
        lang=args.lang,
        alpha_tie=args.alpha_tie,
        macro_tolerance_pp=args.macro_tolerance_pp,
        pct_tolerance_pp=args.pct_tolerance_pp,
        label=args.latex_label,
    )
    latex_path = outdir / f"{lang_slug}_rejection_f1_table.tex"
    latex_path.write_text(latex_table, encoding="utf-8")

    print(f"[ok] Wrote LaTeX table: {latex_path}")
    print(f"[ok] Wrote summary CSV: {summary_csv}")

    print("\n=== Rejection count summary ===")
    for block_name, aggregate in aggregates.items():
        if len(aggregates) > 1:
            print(f"\n[{block_name}]")
        a_rej = aggregate["a_rej"]
        a_inc = aggregate["a_inc"]
        for method_idx, method in enumerate(methods):
            disp = METHOD_DISPLAY_NAMES.get(method, method)
            print(f"\n{disp}:")
            for rate_idx, rate in enumerate(rejection_rates):
                r_mean = np.nanmean(a_rej[method_idx, :, rate_idx])
                r_std = np.nanstd(a_rej[method_idx, :, rate_idx])
                i_mean = np.nanmean(a_inc[method_idx, :, rate_idx])
                i_std = np.nanstd(a_inc[method_idx, :, rate_idx])
                print(
                    f"  {int(rate * 100)}% rejected: {r_mean:.1f} ± {r_std:.1f}   "
                    f"incorrect: {i_mean:.1f} ± {i_std:.1f}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
