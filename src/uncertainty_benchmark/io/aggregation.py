"""Aggregation utilities for fold-level results."""

from __future__ import annotations

import pandas as pd


def summarise_numeric_columns(df: pd.DataFrame, id_columns=None) -> pd.DataFrame:
    """Return mean/std/min/max summary for numeric columns."""
    if id_columns is None:
        id_columns = []

    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in set(id_columns)
    ]

    if not numeric_cols:
        return pd.DataFrame()

    return (
        df[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "column"})
    )


def summarise_method_metric_times(method_times_all: pd.DataFrame) -> pd.DataFrame:
    """Summarise per-method metric computation times."""
    methods = [col for col in method_times_all.columns if col != "fold"]

    if not methods:
        return pd.DataFrame()

    return (
        method_times_all[methods]
        .agg(["mean", "std", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "method"})
    )


def summarise_total_times(total_times_all: pd.DataFrame) -> pd.DataFrame:
    """Summarise standalone per-method total times."""
    return (
        total_times_all.groupby("method")
        .agg(
            folds=("fold", "nunique"),
            n_eval=("n_eval", "mean"),
            uncertainty_mean_s=("uncertainty_s", "mean"),
            uncertainty_std_s=("uncertainty_s", "std"),
            metrics_mean_s=("metrics_s", "mean"),
            metrics_std_s=("metrics_s", "std"),
            total_mean_s=("total_s", "mean"),
            total_std_s=("total_s", "std"),
            ms_per_ex_mean=("total_ms_per_ex", "mean"),
            ms_per_ex_std=("total_ms_per_ex", "std"),
            ex_per_s_mean=("ex_per_s", "mean"),
            ex_per_s_std=("ex_per_s", "std"),
        )
        .reset_index()
    )
