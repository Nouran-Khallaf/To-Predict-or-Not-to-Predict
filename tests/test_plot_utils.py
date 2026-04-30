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
