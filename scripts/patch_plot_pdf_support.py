#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patch plotting utilities to support saving figures as PNG and PDF.

Run from repo root:

    python scripts/patch_plot_pdf_support.py
    pytest tests/test_plot_utils.py
"""

from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def write(path: str, content: str):
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    print(f"[write] {out}")


def main():
    write(
        "src/uncertainty_benchmark/visualisation/plots.py",
        '''
        """Plotting utilities for uncertainty benchmark outputs."""

        from __future__ import annotations

        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd


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


        def normalise_formats(formats=None) -> list[str]:
            """Normalise requested figure formats."""
            if formats is None:
                return ["png"]

            if isinstance(formats, str):
                formats = [formats]

            out = []
            for fmt in formats:
                fmt = str(fmt).lower().lstrip(".").strip()
                if fmt:
                    out.append(fmt)

            if not out:
                out = ["png"]

            return out


        def save_current_figure(path, dpi: int = 300, formats=None):
            """Save current matplotlib figure in one or more formats.

            Parameters
            ----------
            path:
                Output path. The suffix is replaced by each requested format.

            dpi:
                Resolution for raster outputs such as PNG.

            formats:
                List of formats, e.g. ["png", "pdf"].

            Returns
            -------
            dict
                Mapping from format to saved path.
            """
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            formats = normalise_formats(formats)

            plt.tight_layout()

            saved = {}
            for fmt in formats:
                out_path = path.with_suffix(f".{fmt}")

                if fmt == "pdf":
                    plt.savefig(out_path, bbox_inches="tight")
                else:
                    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")

                saved[fmt] = out_path

            plt.close()
            return saved


        def order_methods(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
            out = df.copy()
            order = {method: i for i, method in enumerate(DEFAULT_METHOD_ORDER)}
            out["_method_order"] = out[method_col].map(order).fillna(999)
            out = out.sort_values(["_method_order", method_col]).drop(columns=["_method_order"])
            return out


        def order_metrics(df: pd.DataFrame, metric_col: str = "metric") -> pd.DataFrame:
            out = df.copy()
            order = {metric: i for i, metric in enumerate(DEFAULT_METRIC_ORDER)}
            out["_metric_order"] = out[metric_col].map(order).fillna(999)
            out = out.sort_values(["_metric_order", metric_col]).drop(columns=["_metric_order"])
            return out


        def plot_metric_summary(
            metrics_summary: pd.DataFrame,
            metric: str,
            output_path,
            value_col: str = "mean",
            error_col: str = "std",
            title: str | None = None,
            dpi: int = 300,
            formats=None,
        ):
            """Plot mean metric value by uncertainty method."""
            required = {"method", "metric", value_col}
            missing = required.difference(metrics_summary.columns)
            if missing:
                raise KeyError(f"Missing required columns: {sorted(missing)}")

            df = metrics_summary[metrics_summary["metric"] == metric].copy()
            df = order_methods(df)

            if df.empty:
                raise ValueError(f"No rows found for metric: {metric}")

            x = np.arange(len(df))
            y = df[value_col].astype(float).values

            plt.figure(figsize=(max(7, len(df) * 0.75), 4.5))

            if error_col in df.columns:
                yerr = df[error_col].astype(float).values
                plt.bar(x, y, yerr=yerr, capsize=3)
            else:
                plt.bar(x, y)

            plt.xticks(x, df["method"], rotation=45, ha="right")
            plt.ylabel(metric)
            plt.xlabel("Uncertainty method")
            plt.title(title or f"{metric} by uncertainty method")
            plt.grid(axis="y", alpha=0.3)

            return save_current_figure(output_path, dpi=dpi, formats=formats)


        def plot_timing_summary(
            timing_summary: pd.DataFrame,
            output_path,
            value_col: str = "total_mean_s",
            error_col: str = "total_std_s",
            title: str | None = None,
            dpi: int = 300,
            formats=None,
        ):
            """Plot standalone timing by method."""
            required = {"method", value_col}
            missing = required.difference(timing_summary.columns)
            if missing:
                raise KeyError(f"Missing required columns: {sorted(missing)}")

            df = order_methods(timing_summary.copy())

            x = np.arange(len(df))
            y = df[value_col].astype(float).values

            plt.figure(figsize=(max(7, len(df) * 0.75), 4.5))

            if error_col in df.columns:
                yerr = df[error_col].astype(float).values
                plt.bar(x, y, yerr=yerr, capsize=3)
            else:
                plt.bar(x, y)

            plt.xticks(x, df["method"], rotation=45, ha="right")
            plt.ylabel("Time")
            plt.xlabel("Uncertainty method")
            plt.title(title or "Standalone total time by uncertainty method")
            plt.grid(axis="y", alpha=0.3)

            return save_current_figure(output_path, dpi=dpi, formats=formats)


        def plot_ms_per_example(
            timing_summary: pd.DataFrame,
            output_path,
            title: str | None = None,
            dpi: int = 300,
            formats=None,
        ):
            """Plot milliseconds per example by method."""
            return plot_timing_summary(
                timing_summary=timing_summary,
                output_path=output_path,
                value_col="ms_per_ex_mean",
                error_col="ms_per_ex_std",
                title=title or "Milliseconds per example by uncertainty method",
                dpi=dpi,
                formats=formats,
            )


        def plot_metric_heatmap(
            metrics_summary: pd.DataFrame,
            output_path,
            value_col: str = "mean",
            title: str | None = None,
            dpi: int = 300,
            formats=None,
        ):
            """Plot a compact heatmap of metric means by method."""
            required = {"method", "metric", value_col}
            missing = required.difference(metrics_summary.columns)
            if missing:
                raise KeyError(f"Missing required columns: {sorted(missing)}")

            df = metrics_summary[["method", "metric", value_col]].copy()
            df = order_methods(order_metrics(df))

            matrix = df.pivot(index="method", columns="metric", values=value_col)

            metric_cols = [m for m in DEFAULT_METRIC_ORDER if m in matrix.columns]
            extra_cols = [c for c in matrix.columns if c not in metric_cols]
            matrix = matrix[metric_cols + extra_cols]

            plt.figure(figsize=(max(8, len(matrix.columns) * 0.9), max(4, len(matrix) * 0.45)))
            plt.imshow(matrix.values, aspect="auto")

            plt.xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
            plt.yticks(np.arange(len(matrix.index)), matrix.index)

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix.values[i, j]
                    if pd.notna(value):
                        plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

            plt.colorbar(label=value_col)
            plt.title(title or "Metric summary heatmap")
            plt.xlabel("Metric")
            plt.ylabel("Method")

            return save_current_figure(output_path, dpi=dpi, formats=formats)


        def plot_uncertainty_score_distribution(
            scores_long: pd.DataFrame,
            output_path,
            methods: list[str] | None = None,
            bins: int = 30,
            title: str | None = None,
            dpi: int = 300,
            formats=None,
        ):
            """Plot uncertainty score distributions by method."""
            required = {"method", "uncertainty_score"}
            missing = required.difference(scores_long.columns)
            if missing:
                raise KeyError(f"Missing required columns: {sorted(missing)}")

            df = scores_long.copy()

            if methods is not None:
                df = df[df["method"].isin(methods)].copy()

            df = order_methods(df)

            method_list = list(dict.fromkeys(df["method"].tolist()))

            if not method_list:
                raise ValueError("No methods available for score distribution plot.")

            plt.figure(figsize=(max(7, len(method_list) * 0.8), 4.5))

            for method in method_list:
                vals = df[df["method"] == method]["uncertainty_score"].astype(float).values
                plt.hist(vals, bins=bins, alpha=0.35, label=method)

            plt.xlabel("Uncertainty score")
            plt.ylabel("Count")
            plt.title(title or "Uncertainty score distributions")
            plt.legend()
            plt.grid(axis="y", alpha=0.3)

            return save_current_figure(output_path, dpi=dpi, formats=formats)


        def make_all_figures(results_dir, output_dir=None, dpi: int = 300, formats=None) -> dict:
            """Create standard figures from a completed results directory.

            Use formats=["png", "pdf"] to save both.
            """
            formats = normalise_formats(formats)
            results_dir = Path(results_dir)

            if output_dir is None:
                output_dir = results_dir / "figures"

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

            outputs = {}

            for metric in ["ECE", "AU-PRC", "ROC-AUC", "TI@95"]:
                if metric in set(metrics_summary["metric"]):
                    safe_metric = metric.replace("@", "at").replace("/", "_")
                    outputs[f"metric_{metric}"] = plot_metric_summary(
                        metrics_summary,
                        metric=metric,
                        output_path=output_dir / f"metric_{safe_metric}.png",
                        dpi=dpi,
                        formats=formats,
                    )

            outputs["timing_total"] = plot_timing_summary(
                timing_summary,
                output_path=output_dir / "timing_total_seconds.png",
                dpi=dpi,
                formats=formats,
            )

            outputs["timing_ms_per_example"] = plot_ms_per_example(
                timing_summary,
                output_path=output_dir / "timing_ms_per_example.png",
                dpi=dpi,
                formats=formats,
            )

            outputs["metric_heatmap"] = plot_metric_heatmap(
                metrics_summary,
                output_path=output_dir / "metric_heatmap.png",
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
        ''',
    )

    write(
        "scripts/make_figures.py",
        '''
        #!/usr/bin/env python3
        # -*- coding: utf-8 -*-

        """Create standard figures from benchmark outputs."""

        import argparse

        from uncertainty_benchmark.visualisation import make_all_figures


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--results-dir",
                required=True,
                help="Path to a completed experiment results directory.",
            )
            parser.add_argument(
                "--outdir",
                default=None,
                help="Output directory for figures. Defaults to results-dir/figures.",
            )
            parser.add_argument(
                "--dpi",
                type=int,
                default=300,
                help="Figure resolution for raster formats such as PNG.",
            )
            parser.add_argument(
                "--formats",
                nargs="+",
                default=["png"],
                help="Figure formats to save, e.g. --formats png pdf",
            )
            args = parser.parse_args()

            outputs = make_all_figures(
                results_dir=args.results_dir,
                output_dir=args.outdir,
                dpi=args.dpi,
                formats=args.formats,
            )

            print("Saved figures:")
            for name, paths in outputs.items():
                print(f"  {name}:")
                if isinstance(paths, dict):
                    for fmt, path in paths.items():
                        print(f"    {fmt}: {path}")
                else:
                    print(f"    {paths}")


        if __name__ == "__main__":
            main()
        ''',
    )

    write(
        "tests/test_plot_utils.py",
        '''
        import pandas as pd

        from uncertainty_benchmark.visualisation.plots import (
            make_all_figures,
            normalise_formats,
            plot_metric_heatmap,
            plot_metric_summary,
            plot_ms_per_example,
            plot_timing_summary,
            plot_uncertainty_score_distribution,
        )


        def sample_metrics_summary():
            return pd.DataFrame(
                {
                    "method": ["SR", "SR", "ENT", "ENT"],
                    "metric": ["ECE", "ROC-AUC", "ECE", "ROC-AUC"],
                    "folds": [2, 2, 2, 2],
                    "mean": [0.1, 0.8, 0.2, 0.85],
                    "std": [0.01, 0.02, 0.03, 0.04],
                    "min": [0.09, 0.78, 0.17, 0.81],
                    "max": [0.11, 0.82, 0.23, 0.89],
                }
            )


        def sample_timing_summary():
            return pd.DataFrame(
                {
                    "method": ["SR", "ENT"],
                    "folds": [2, 2],
                    "n_eval": [100, 100],
                    "uncertainty_mean_s": [1.0, 1.1],
                    "uncertainty_std_s": [0.1, 0.2],
                    "metrics_mean_s": [0.01, 0.02],
                    "metrics_std_s": [0.001, 0.002],
                    "total_mean_s": [1.01, 1.12],
                    "total_std_s": [0.1, 0.2],
                    "ms_per_ex_mean": [10.1, 11.2],
                    "ms_per_ex_std": [1.0, 2.0],
                    "ex_per_s_mean": [99.0, 89.0],
                    "ex_per_s_std": [2.0, 3.0],
                }
            )


        def test_normalise_formats():
            assert normalise_formats(None) == ["png"]
            assert normalise_formats("pdf") == ["pdf"]
            assert normalise_formats(["png", ".pdf"]) == ["png", "pdf"]


        def test_plot_metric_summary_saves_png_and_pdf(tmp_path):
            out = plot_metric_summary(
                sample_metrics_summary(),
                metric="ECE",
                output_path=tmp_path / "metric_ece.png",
                formats=["png", "pdf"],
            )

            assert out["png"].exists()
            assert out["pdf"].exists()


        def test_plot_timing_summary_saves_png_and_pdf(tmp_path):
            out = plot_timing_summary(
                sample_timing_summary(),
                output_path=tmp_path / "timing.png",
                formats=["png", "pdf"],
            )

            assert out["png"].exists()
            assert out["pdf"].exists()


        def test_plot_ms_per_example(tmp_path):
            out = plot_ms_per_example(
                sample_timing_summary(),
                output_path=tmp_path / "ms_per_example.png",
                formats=["png", "pdf"],
            )

            assert out["png"].exists()
            assert out["pdf"].exists()


        def test_plot_metric_heatmap(tmp_path):
            out = plot_metric_heatmap(
                sample_metrics_summary(),
                output_path=tmp_path / "heatmap.png",
                formats=["png", "pdf"],
            )

            assert out["png"].exists()
            assert out["pdf"].exists()


        def test_plot_uncertainty_score_distribution(tmp_path):
            scores = pd.DataFrame(
                {
                    "method": ["SR", "SR", "ENT", "ENT"],
                    "uncertainty_score": [0.1, 0.2, 0.3, 0.4],
                }
            )

            out = plot_uncertainty_score_distribution(
                scores,
                output_path=tmp_path / "dist.png",
                formats=["png", "pdf"],
            )

            assert out["png"].exists()
            assert out["pdf"].exists()


        def test_make_all_figures(tmp_path):
            results_dir = tmp_path / "results"
            metrics_dir = results_dir / "metrics"
            timing_dir = results_dir / "timing"
            scores_dir = results_dir / "scores"

            metrics_dir.mkdir(parents=True)
            timing_dir.mkdir(parents=True)
            scores_dir.mkdir(parents=True)

            sample_metrics_summary().to_csv(
                metrics_dir / "metrics_summary_mean_std.csv",
                index=False,
            )

            sample_timing_summary().to_csv(
                timing_dir / "method_total_times_summary_mean_std.csv",
                index=False,
            )

            pd.DataFrame(
                {
                    "fold": [0, 0, 0, 0],
                    "method": ["SR", "SR", "ENT", "ENT"],
                    "uncertainty_score": [0.1, 0.2, 0.3, 0.4],
                }
            ).to_csv(scores_dir / "fold_0_scores_long.csv", index=False)

            outputs = make_all_figures(results_dir, formats=["png", "pdf"])

            assert "timing_total" in outputs
            assert "metric_heatmap" in outputs
            assert outputs["timing_total"]["png"].exists()
            assert outputs["timing_total"]["pdf"].exists()
            assert outputs["metric_heatmap"]["png"].exists()
            assert outputs["metric_heatmap"]["pdf"].exists()
        ''',
    )

    print("\\nDone. PDF figure support added.")
    print("\\nNow run:")
    print("  pytest tests/test_plot_utils.py")
    print("  pytest tests")


if __name__ == "__main__":
    main()