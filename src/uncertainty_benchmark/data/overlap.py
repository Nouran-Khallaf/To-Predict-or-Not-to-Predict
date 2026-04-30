"""Utilities for removing train/eval text overlap."""

from __future__ import annotations

import pandas as pd


def normalise_text_series(series: pd.Series) -> pd.Series:
    """Normalise text for overlap checks."""
    return series.dropna().astype(str).str.strip()


def remove_text_overlap(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_text_col: str = "text",
    eval_text_col: str = "text",
) -> pd.DataFrame:
    """Remove rows from train_df whose text appears in eval_df."""
    if train_text_col not in train_df.columns:
        raise KeyError(f"Missing train text column: {train_text_col}")

    if eval_text_col not in eval_df.columns:
        raise KeyError(f"Missing eval text column: {eval_text_col}")

    eval_texts = set(normalise_text_series(eval_df[eval_text_col]))
    train_texts = train_df[train_text_col].astype(str).str.strip()

    return train_df[~train_texts.isin(eval_texts)].copy().reset_index(drop=True)


def remove_predicted_rows_by_file(
    lang_df: pd.DataFrame,
    source_path: str,
    pred_file_path: str,
    text_col: str = "Sentence",
) -> pd.DataFrame:
    """Compatibility function from the original script.

    This removes rows from a raw language dataframe if they occur in the
    prediction CSV for the corresponding source filename.
    """
    from pathlib import Path

    df_pred = pd.read_csv(pred_file_path)
    lang_name = Path(source_path).stem

    if "Lang" not in df_pred.columns:
        raise KeyError("Prediction CSV must contain a 'Lang' column.")

    df_pred = df_pred[df_pred["Lang"] == lang_name].copy()

    pred_text_col = (
        text_col
        if text_col in df_pred.columns
        else "Sentence"
        if "Sentence" in df_pred.columns
        else "text"
        if "text" in df_pred.columns
        else None
    )

    if pred_text_col is None:
        raise KeyError("Could not find a text column in predictions.")

    pred_texts = set(normalise_text_series(df_pred[pred_text_col]))
    return lang_df[
        ~lang_df[text_col].astype(str).str.strip().isin(pred_texts)
    ].copy()
