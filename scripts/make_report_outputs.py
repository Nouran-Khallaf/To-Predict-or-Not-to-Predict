#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create standard reporting outputs from uncertainty benchmark summaries.

Run from the repo root, for example:

    python scripts/make_report_outputs.py --config configs/reporting_english_validation_mbert.yaml

The script expects a long CSV after column mapping with at least:

    method, metric, mean, std

It also supports evaluator outputs that use ``mode`` instead of ``method``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from uncertainty_benchmark.reporting.latex_tables import build_metric_summary_table
from uncertainty_benchmark.reporting.plots import plot_metric_by_method, plot_metric_heatmap


DEFAULT_METHOD_ORDER = [
    "SR",
    "SMP",
    "ENT",
    "PV",
    "BALD",
    "ENT_MC",
    "MD",
    "HUQ-MD",
    "LOF",
    "ISOF",
]

# This is the original selective-prediction set from your earlier table/text.
DEFAULT_REPORT_METRICS = [
    "RC-AUC",
    "Norm RC-AUC",
    "E-AUoptRC",
    "TI",
    "TI@95",
    "Optimal Coverage",
]

# Metrics where smaller is better.
DEFAULT_LOWER_IS_BETTER = {
    "ECE",
    "RC-AUC",
    "Norm RC-AUC",
    "E-AUoptRC",
}

# Metrics with an ideal target value rather than simple min/max.
DEFAULT_TARGET_IS_BEST = {
    "CITL": 0.0,
    "C-Slope": 1.0,
}

COMMON_COLUMN_RENAMES = {
    "mode": "method",
    "Mode": "method",
    "Method": "method",
    "measure": "metric",
    "Metric": "metric",
    "Mean": "mean",
    "Std": "std",
    "STD": "std",
}


def _safe_name(text: str) -> str:
    """Make a metric/language name safe for filenames."""
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


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config if supplied."""
    if path is None:
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("YAML config support requires PyYAML: pip install pyyaml") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping/dictionary: {config_path}")
    return config


def cfg_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read nested config values using dot notation."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def cli_or_config(value: Any, config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Prefer explicit CLI value, then config value, then default."""
    if value is not None:
        return value
    return cfg_get(config, dotted_key, default)


def normalise_list(value: Any, default: Sequence[str] | None = None) -> list[str]:
    """Convert config/CLI scalar or list values into a list of strings."""
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def find_metrics_summary(results_dir: Path, explicit_path: str | Path | None = None) -> Path:
    """Find the metrics summary CSV."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"metrics summary not found: {path}")
        return path

    candidates = [
        results_dir / "metrics" / "metrics_summary_mean_std.csv",
        results_dir / "metrics_summary_mean_std.csv",
        results_dir / "metrics" / "metrics_summary.csv",
        results_dir / "metrics_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    globbed = sorted(results_dir.glob("**/*metrics_summary*mean*std*.csv"))
    if globbed:
        return globbed[0]

    globbed = sorted(results_dir.glob("**/*metrics_summary*.csv"))
    if globbed:
        return globbed[0]

    raise FileNotFoundError(
        "Could not find a metrics summary CSV. Expected something like "
        f"{results_dir / 'metrics' / 'metrics_summary_mean_std.csv'}"
    )


def apply_column_mapping(df: pd.DataFrame, column_map: Mapping[str, str] | None) -> pd.DataFrame:
    """Rename input CSV columns to the standard reporting names.

    Config format is standard_name: source_column_name, for example:

        method: mode
    """
    df = df.copy()

    if column_map:
        rename_map = {}
        for standard_name, source_name in column_map.items():
            if source_name in df.columns and source_name != standard_name:
                rename_map[source_name] = standard_name
        df = df.rename(columns=rename_map)

    df = df.rename(columns={k: v for k, v in COMMON_COLUMN_RENAMES.items() if k in df.columns})
    return df


def load_long_summary(path: Path, column_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Load and validate a long-format metric summary."""
    df = pd.read_csv(path)
    df = apply_column_mapping(df, column_map)

    required = {"method", "metric", "mean", "std"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns {sorted(missing)} after column mapping. "
            f"Available columns: {list(df.columns)}. Expected: method, metric, mean, std."
        )

    df = df.copy()
    df["method"] = df["method"].astype(str)
    df["metric"] = df["metric"].astype(str)
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    return df


def filter_and_order_summary(
    summary: pd.DataFrame,
    methods: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Filter to requested methods/metrics and preserve the requested order."""
    df = summary.copy()

    if methods:
        method_set = set(methods)
        df = df[df["method"].isin(method_set)].copy()
        df["method"] = pd.Categorical(df["method"], categories=list(methods), ordered=True)

    if metrics:
        metric_set = set(metrics)
        df = df[df["metric"].isin(metric_set)].copy()
        df["metric"] = pd.Categorical(df["metric"], categories=list(metrics), ordered=True)

    sort_cols = []
    if "metric" in df.columns:
        sort_cols.append("metric")
    if "method" in df.columns:
        sort_cols.append("method")
    if sort_cols:
        df = df.sort_values(sort_cols)

    df["method"] = df["method"].astype(str)
    df["metric"] = df["metric"].astype(str)
    return df


def make_wide_tables(
    summary: pd.DataFrame,
    methods: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert long summary to metric x method mean/std tables."""
    mean_values = summary.pivot_table(index="metric", columns="method", values="mean", aggfunc="first")
    std_values = summary.pivot_table(index="metric", columns="method", values="std", aggfunc="first")

    if metrics:
        present_metrics = [m for m in metrics if m in mean_values.index]
        mean_values = mean_values.reindex(present_metrics)
        std_values = std_values.reindex(present_metrics)

    if methods:
        present_methods = [m for m in methods if m in mean_values.columns]
        mean_values = mean_values.reindex(columns=present_methods)
        std_values = std_values.reindex(columns=present_methods)

    return mean_values, std_values


def choose_best_methods(
    mean_values: pd.DataFrame,
    lower_is_better: set[str] | None = None,
    target_is_best: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Choose best method per metric using metric-specific direction rules."""
    lower_is_better = lower_is_better or DEFAULT_LOWER_IS_BETTER
    target_is_best = target_is_best or DEFAULT_TARGET_IS_BEST

    best: dict[str, str] = {}
    for metric in mean_values.index:
        vals = pd.to_numeric(mean_values.loc[metric], errors="coerce").dropna()
        if vals.empty:
            continue

        metric_name = str(metric)
        if metric_name in target_is_best:
            target = float(target_is_best[metric_name])
            best[metric_name] = str((vals - target).abs().idxmin())
        elif metric_name in lower_is_better:
            best[metric_name] = str(vals.idxmin())
        else:
            best[metric_name] = str(vals.idxmax())

    return best


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config file for reporting.")
    parser.add_argument("--results-dir", default=None, help="Experiment results directory.")
    parser.add_argument("--metrics-summary", default=None, help="Optional explicit metrics summary CSV path.")
    parser.add_argument("--lang", default=None, help="Language/corpus label for captions and filenames, e.g. EN.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Default: <results-dir>/reporting")
    parser.add_argument("--methods", nargs="*", default=None, help="Methods to keep, in reporting order.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Metrics to keep, in reporting order.")
    parser.add_argument("--metrics-to-plot", nargs="*", default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--formats", nargs="*", default=None, help="Figure formats, e.g. png pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    results_dir_value = cli_or_config(args.results_dir, config, "paths.results_dir", None)
    if not results_dir_value:
        raise ValueError("Please provide --results-dir or set paths.results_dir in the config.")

    results_dir = Path(results_dir_value)
    metrics_summary = cli_or_config(args.metrics_summary, config, "paths.metrics_summary", None)
    output_dir_value = cli_or_config(args.output_dir, config, "paths.output_dir", None)
    output_dir = Path(output_dir_value) if output_dir_value else results_dir / "reporting"

    lang = cli_or_config(args.lang, config, "lang", cfg_get(config, "language", ""))
    lang_safe = _safe_name(str(lang).lower()) if lang else "overall"

    methods = normalise_list(
        cli_or_config(args.methods, config, "filters.methods", None),
        default=DEFAULT_METHOD_ORDER,
    )
    metrics = normalise_list(
        cli_or_config(args.metrics, config, "filters.metrics", None),
        default=DEFAULT_REPORT_METRICS,
    )

    metrics_to_plot = normalise_list(
        cli_or_config(args.metrics_to_plot, config, "plots.metrics_to_plot", None),
        default=metrics,
    )
    formats = normalise_list(cli_or_config(args.formats, config, "plots.formats", ["png"]))
    dpi = int(cli_or_config(args.dpi, config, "plots.dpi", 300))

    make_tables = bool(cfg_get(config, "tables.enabled", True))
    make_plots = bool(cfg_get(config, "plots.enabled", True))
    make_heatmap = bool(cfg_get(config, "plots.heatmap", True))
    write_wide_mean = bool(cfg_get(config, "outputs.write_wide_mean_csv", True))
    write_wide_std = bool(cfg_get(config, "outputs.write_wide_std_csv", False))
    write_filtered_long = bool(cfg_get(config, "outputs.write_filtered_long_csv", True))

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"

    metrics_path = find_metrics_summary(results_dir, metrics_summary)
    column_map = cfg_get(config, "input.column_map", None)
    summary = load_long_summary(metrics_path, column_map=column_map)
    summary = filter_and_order_summary(summary, methods=methods, metrics=metrics)

    if summary.empty:
        raise ValueError(
            "No rows left after applying reporting filters. Check filters.methods and filters.metrics in the config."
        )

    mean_values, std_values = make_wide_tables(summary, methods=methods, metrics=metrics)

    lower_is_better = set(cfg_get(config, "metric_rules.lower_is_better", list(DEFAULT_LOWER_IS_BETTER)))
    target_is_best = cfg_get(config, "metric_rules.target_is_best", DEFAULT_TARGET_IS_BEST)
    best_methods = choose_best_methods(
        mean_values=mean_values,
        lower_is_better=lower_is_better,
        target_is_best=target_is_best,
    )

    print(f"Read metrics summary: {metrics_path}")
    print(f"Using methods: {', '.join(mean_values.columns)}")
    print(f"Using metrics: {', '.join(mean_values.index)}")

    if write_filtered_long:
        filename = cfg_get(config, "outputs.filtered_long_csv_filename", f"metrics_summary_filtered_{lang_safe}.csv")
        path = csv_dir / filename
        summary.to_csv(path, index=False)
        print(f"Wrote filtered long CSV: {path}")

    if write_wide_mean:
        filename = cfg_get(config, "outputs.wide_mean_csv_filename", f"metrics_summary_mean_wide_{lang_safe}.csv")
        path = csv_dir / filename
        write_csv(path, mean_values)
        print(f"Wrote wide mean CSV: {path}")

    if write_wide_std:
        filename = cfg_get(config, "outputs.wide_std_csv_filename", f"metrics_summary_std_wide_{lang_safe}.csv")
        path = csv_dir / filename
        write_csv(path, std_values)
        print(f"Wrote wide std CSV: {path}")

    if make_tables:
        table_filename = cfg_get(config, "tables.filename", f"metrics_summary_{lang_safe}.tex")
        table_label = cfg_get(config, "tables.label", f"tab:metrics_summary_{lang_safe}")
        table_caption = cfg_get(config, "tables.caption", None)
        table_tex = build_metric_summary_table(
            mean_values=mean_values,
            std_values=std_values,
            best_methods=best_methods,
            close_methods={},
            lang=lang,
            label=table_label,
            caption=table_caption,
        )
        table_path = tables_dir / table_filename
        write_text(table_path, table_tex)
        print(f"Wrote LaTeX table: {table_path}")
    else:
        print("Tables disabled in config.")

    written_figures: list[Path] = []
    if make_plots:
        for metric in metrics_to_plot:
            if metric not in mean_values.index:
                print(f"Skipping missing metric: {metric}")
                continue
            for fmt in formats:
                fmt = str(fmt).lower().lstrip(".")
                outpath = figures_dir / f"metric_{_safe_name(metric)}_{lang_safe}.{fmt}"
                plot_metric_by_method(
                    mean_values=mean_values,
                    metric=metric,
                    methods=list(mean_values.columns),
                    title=f"{metric} by uncertainty method" if not lang else f"{lang}: {metric} by uncertainty method",
                    outpath=outpath,
                    dpi=dpi,
                    close=True,
                )
                written_figures.append(outpath)

        if make_heatmap:
            heatmap_input = summary[["method", "metric", "mean"]].copy()
            for fmt in formats:
                fmt = str(fmt).lower().lstrip(".")
                outpath = figures_dir / f"metrics_heatmap_{lang_safe}.{fmt}"
                plot_metric_heatmap(
                    heatmap_input,
                    output_path=outpath,
                    value_col="mean",
                    title=f"{lang}: selective metrics by method" if lang else "Selective metrics by method",
                    dpi=dpi,
                    formats=[fmt],
                )
                written_figures.append(outpath)

        if written_figures:
            print("Wrote figures:")
            for path in written_figures:
                print(f"  {path}")
        else:
            print("No figures written because none of the requested metrics were found.")
    else:
        print("Plots disabled in config.")


if __name__ == "__main__":
    main()
