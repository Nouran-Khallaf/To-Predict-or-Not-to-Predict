import pandas as pd

from uncertainty_benchmark.io.tables import (
    build_metrics_paper_table,
    build_timing_paper_table,
    create_paper_tables,
    format_mean_std,
)


def test_format_mean_std():
    assert format_mean_std(1.23456, 0.12345, decimals=2) == "1.23 ± 0.12"
    assert format_mean_std(1.23456, None, decimals=2) == "1.23"


def test_build_metrics_paper_table():
    metrics_summary = pd.DataFrame(
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

    table = build_metrics_paper_table(metrics_summary, decimals=2)

    assert "Method" in table.columns
    assert "Method ID" in table.columns
    assert "ECE" in table.columns
    assert "ROC-AUC" in table.columns
    assert len(table) == 2


def test_build_timing_paper_table():
    timing_summary = pd.DataFrame(
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

    table = build_timing_paper_table(timing_summary, decimals=2)

    assert "Method" in table.columns
    assert "Total time (s)" in table.columns
    assert "ms / example" in table.columns
    assert len(table) == 2


def test_create_paper_tables(tmp_path):
    results_dir = tmp_path / "results"
    metrics_dir = results_dir / "metrics"
    timing_dir = results_dir / "timing"
    metrics_dir.mkdir(parents=True)
    timing_dir.mkdir(parents=True)

    metrics_summary = pd.DataFrame(
        {
            "method": ["SR", "SR"],
            "metric": ["ECE", "ROC-AUC"],
            "folds": [1, 1],
            "mean": [0.1, 0.8],
            "std": [0.0, 0.0],
            "min": [0.1, 0.8],
            "max": [0.1, 0.8],
        }
    )
    metrics_summary.to_csv(metrics_dir / "metrics_summary_mean_std.csv", index=False)

    timing_summary = pd.DataFrame(
        {
            "method": ["SR"],
            "folds": [1],
            "n_eval": [100],
            "uncertainty_mean_s": [1.0],
            "uncertainty_std_s": [0.0],
            "metrics_mean_s": [0.01],
            "metrics_std_s": [0.0],
            "total_mean_s": [1.01],
            "total_std_s": [0.0],
            "ms_per_ex_mean": [10.1],
            "ms_per_ex_std": [0.0],
            "ex_per_s_mean": [99.0],
            "ex_per_s_std": [0.0],
        }
    )
    timing_summary.to_csv(
        timing_dir / "method_total_times_summary_mean_std.csv",
        index=False,
    )

    outputs = create_paper_tables(results_dir)

    assert outputs["metrics_csv"].exists()
    assert outputs["metrics_tex"].exists()
    assert outputs["timing_csv"].exists()
    assert outputs["timing_tex"].exists()
