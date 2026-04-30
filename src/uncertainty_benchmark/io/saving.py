"""Output saving utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def scores_wide_to_long(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Convert per-sample wide score table to tidy long format."""
    id_cols = [
        col for col in [
            "fold",
            "text",
            "y_true_idx",
            "y_pred_idx",
            "true_label",
            "predicted_label",
            "correct",
        ]
        if col in df.columns
    ]

    return df.melt(
        id_vars=id_cols,
        value_vars=methods,
        var_name="method",
        value_name="uncertainty_score",
    )
