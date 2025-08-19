# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

# ---------------- CONFIG ----------------
LANG_GLOBS = {
    "AR": "ar_fold*_metrics_summary*.csv",
    "EN": "en_fold*_metrics_summary*.csv",
    "FR": "fr_fold*_metrics_summary*.csv",
    "HI": "hi_fold*_metrics_summary*.csv",
    "RU": "ru_fold*_metrics_summary*.csv",
}

# Metric pairs by group 
metric_pairs_by_group = {
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

# variation aliases to match CSV metric names
METRIC_ALIASES = {
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

OUT_DIR = "tables"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def _first_present(canonical, index_like):
    aliases = METRIC_ALIASES.get(canonical, [canonical])
    lowmap = {s.lower(): s for s in index_like}
    for a in aliases:
        if a.lower() in lowmap:
            return lowmap[a.lower()]
    return None

def load_language_folds(glob_pat):
    files = sorted(glob.glob(glob_pat))
    if not files:
        return []
    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce")
        # Drop duplicate metric rows, keep first (fixes repeated ROC-AUC)
        df = df[~df.index.duplicated(keep="first")]
        dfs.append(df)
    return dfs  # list of DataFrames (rows=metrics, cols=methods), one per fold

def concat_metric_vector_across_folds_and_methods(fold_dfs, metric_name):
    """
    Build a long vector for one metric by concatenating (in fold order)
    the per-method values (in a consistent method order across folds).
    """
    if not fold_dfs:
        return None

    # Methods common to ALL folds, keep their order from the first fold
    first_methods = list(fold_dfs[0].columns)
    common_methods = [m for m in first_methods if all(m in df.columns for df in fold_dfs)]
    if len(common_methods) < 2:
        return None  # need at least two methods to correlate

    # Resolve metric row per fold (using aliases), then concatenate
    series_list = []
    for df in fold_dfs:
        rowname = _first_present(metric_name, df.index)
        if rowname is None or rowname not in df.index:
            return None  # this fold/language lacks the metric entirely
        s = df.loc[rowname, common_methods]
        series_list.append(s)

    vec = pd.concat(series_list, axis=0)  # length = n_folds * n_methods
    return vec

def compute_corr_for_pair_per_language(lang2folds, m1, m2):
    """
    For each language: build concatenated vectors for m1 and m2 and compute:
      - Kendall's tau (with p-value)
      - Pearson's r (plain)
    Returns: tau_vals (dict), tau_p (dict), r_vals (dict)
    """
    tau_vals, tau_p, r_vals = {}, {}, {}

    for lang in lang2folds:  # preserves configured order
        folds = lang2folds[lang]
        v1 = concat_metric_vector_across_folds_and_methods(folds, m1)
        v2 = concat_metric_vector_across_folds_and_methods(folds, m2)

        if v1 is None or v2 is None:
            tau_vals[lang] = np.nan
            tau_p[lang] = np.nan
            r_vals[lang] = np.nan
            continue

        mask = ~(v1.isna() | v2.isna())
        x, y = v1[mask].values, v2[mask].values
        if x.size < 2:
            tau_vals[lang] = np.nan; tau_p[lang] = np.nan; r_vals[lang] = np.nan
            continue

        t, p_t = kendalltau(x, y, nan_policy="omit")
        tau_vals[lang] = float(t)
        tau_p[lang]    = float(p_t) if np.isfinite(p_t) else np.nan

        r, _ = pearsonr(x, y)
        r_vals[lang] = float(r)

    return tau_vals, tau_p, r_vals

def make_tables(lang_globs, metric_pairs_by_group):
    # Load folds per language in the configured order
    lang2folds = {lang: load_language_folds(glob_pat) for lang, glob_pat in lang_globs.items()}
    langs = list(lang2folds.keys())

    # Prepare output structures
    tau_tables = {g: pd.DataFrame(index=pd.MultiIndex.from_tuples(pairs, names=["m1","m2"]), columns=langs)
                  for g, pairs in metric_pairs_by_group.items()}
    tau_p_tables = {g: pd.DataFrame(index=pd.MultiIndex.from_tuples(pairs, names=["m1","m2"]), columns=langs)
                    for g, pairs in metric_pairs_by_group.items()}
    r_tables = {g: pd.DataFrame(index=pd.MultiIndex.from_tuples(pairs, names=["m1","m2"]), columns=langs)
                for g, pairs in metric_pairs_by_group.items()}

    # Compute per pair
    for g, pairs in metric_pairs_by_group.items():
        for (m1, m2) in pairs:
            tau_vals, tau_p, r_vals = compute_corr_for_pair_per_language(lang2folds, m1, m2)
            for L in langs:
                tau_tables[g].loc[(m1, m2), L]   = tau_vals.get(L, np.nan)
                tau_p_tables[g].loc[(m1, m2), L] = tau_p.get(L, np.nan)
                r_tables[g].loc[(m1, m2), L]     = r_vals.get(L, np.nan)

    return tau_tables, tau_p_tables, r_tables, langs

# --------------- LaTeX formatting ----------------
def fmt_tau_with_p(tau, p):
    if pd.isna(tau):
        return "—"
    s = f"{tau:.2f}"
    if np.isfinite(p):
        if p < 0.01:
            return f"\\textbf{{{s}}}"
        elif p < 0.05:
            return f"\\underline{{{s}}}"
    return s

def fmt_r(r):
    return "—" if pd.isna(r) else f"{r:.2f}"

def build_tau_table_with_p(tau_tables, tau_p_tables, title_caption, label):
    some_group = next(iter(tau_tables))
    langs = list(tau_tables[some_group].columns)
    header = (
        "\\begin{table}[!t]\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{l " + " ".join(["r"]*len(langs)) + "}\n"
        "\\toprule\n"
        "\\textbf{Metric Pair} & " + " & ".join(f"\\textbf{{{L}}}" for L in langs) + " \\\\\n"
        "\\midrule"
    )
    lines = [header]
    for group in tau_tables:
        tdf, pdf = tau_tables[group], tau_p_tables[group]
        lines.append(f"\\multicolumn{{{len(langs)+1}}}{{l}}{{\\textbf{{{group}}}}} \\\\")
        lines.append("\\midrule")
        for (m1, m2) in tdf.index:
            row = [f"{m1} vs {m2}"]
            for L in langs:
                row.append(fmt_tau_with_p(tdf.loc[(m1, m2), L], pdf.loc[(m1, m2), L]))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    lines.append("\\bottomrule\n\\end{tabular}")
    lines.append(f"\\caption{{{title_caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def build_r_table(r_tables, title_caption, label):
    some_group = next(iter(r_tables))
    langs = list(r_tables[some_group].columns)
    header = (
        "\\begin{table}[!t]\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\centering\n\\small\n"
        "\\begin{tabular}{l " + " ".join(["r"]*len(langs)) + "}\n"
        "\\toprule\n"
        "\\textbf{Metric Pair} & " + " & ".join(f"\\textbf{{{L}}}" for L in langs) + " \\\\\n"
        "\\midrule"
    )
    lines = [header]
    for group, df in r_tables.items():
        lines.append(f"\\multicolumn{{{len(langs)+1}}}{{l}}{{\\textbf{{{group}}}}} \\\\")
        lines.append("\\midrule")
        for (m1, m2), row in df.iterrows():
            cells = [f"{m1} vs {m2}"] + [fmt_r(row[L]) for L in langs]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines.append("\\bottomrule\n\\end{tabular}")
    lines.append(f"\\caption{{{title_caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# ---------------- Run ----------------
tau_tables, tau_p_tables, r_tables, langs = make_tables(LANG_GLOBS, metric_pairs_by_group)

# Kendall τ (p-based formatting)
tau_caption = ("Kendall's correlation ($\\tau$) between metric pairs using concatenated "
               "fold-level rows per language (methods concatenated per fold). "
               "Bold if $p<0.01$, underline if $p<0.05$.")
tau_tex = build_tau_table_with_p(tau_tables, tau_p_tables, tau_caption, "tab:metric_correlations_kendall")
with open(os.path.join(OUT_DIR, "metric_correlations_kendall.tex"), "w") as f:
    f.write(tau_tex)

# Pearson r (plain values)
r_caption = ("Pearson correlation ($r$) between metric pairs using concatenated fold-level rows per language.")
r_tex = build_r_table(r_tables, r_caption, "tab:metric_correlations_pearson")
with open(os.path.join(OUT_DIR, "metric_correlations_pearson.tex"), "w") as f:
    f.write(r_tex)

print("[LaTeX] Wrote:",
      os.path.join(OUT_DIR, "metric_correlations_kendall.tex"),
      "and",
      os.path.join(OUT_DIR, "metric_correlations_pearson.tex"))
