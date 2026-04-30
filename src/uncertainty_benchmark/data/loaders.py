"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset

from uncertainty_benchmark.data.label_mapping import (
    build_label_encoder,
    map_eval_labels,
    map_train_labels,
)


def pick_column(df: pd.DataFrame, candidates, required: bool = True, what: str = "column"):
    """Pick the first available column from a candidate list."""
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(
            f"Could not find {what}. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def read_table(path) -> pd.DataFrame:
    """Read CSV, TSV, Excel, or JSONL into a dataframe."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)

    raise ValueError(
        f"Unsupported file extension for {path}. "
        "Supported: .csv, .tsv, .xlsx, .xls, .jsonl"
    )


def clean_text_column(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Drop empty text rows and standardise text column name to `text`."""
    out = df.copy()
    out[text_col] = out[text_col].astype(str)
    out[text_col] = out[text_col].str.strip()
    out = out[out[text_col] != ""].copy()

    if text_col != "text":
        out = out.rename(columns={text_col: "text"})

    return out


def load_eval_from_prediction_csv(config: dict, fold_id: int, encoder=None) -> pd.DataFrame:
    """Load evaluation subset from a fold prediction CSV.

    Expected prediction CSV columns:
        - Lang
        - Sentence/text
        - True Label/Label
    """
    if encoder is None:
        encoder = build_label_encoder(config["labels"]["classes"])

    pred_template = config["data"]["pred_csv_template"]
    pred_path = pred_template.format(fold_id=fold_id)

    df_pred = read_table(pred_path)

    if "Lang" not in df_pred.columns:
        raise KeyError(
            f"'Lang' column missing in prediction file {pred_path}. "
            f"Available columns: {list(df_pred.columns)}"
        )

    lang_name = config["data"]["lang_name"]
    df_lang = df_pred[df_pred["Lang"] == lang_name].copy()

    if len(df_lang) == 0:
        unique_langs = df_pred["Lang"].dropna().unique()[:20]
        raise ValueError(
            f"No rows found for Lang == {lang_name} in {pred_path}. "
            f"First available Lang values: {unique_langs}"
        )

    text_col = pick_column(
        df_lang,
        config["data"].get(
            "eval_text_column_candidates",
            ["Sentence", "sentence", "text", "Text"],
        ),
        what="evaluation text column",
    )

    label_col = pick_column(
        df_lang,
        config["data"].get(
            "label_column_candidates",
            ["True Label", "True_Label", "TrueLabel", "Label", "gold", "Gold"],
        ),
        what="evaluation label column",
    )

    df_eval = df_lang[[text_col, label_col]].copy()
    df_eval = clean_text_column(df_eval, text_col)
    df_eval = map_eval_labels(df_eval, label_col=label_col, encoder=encoder)

    return df_eval[["text", "labels", "label_text"]].reset_index(drop=True)


def load_train_file(config: dict, encoder=None) -> pd.DataFrame:
    """Load and map the training file.

    Current default assumes:
        - Sentence column for text
        - Rating column for labels

    More general column candidates are still supported.
    """
    if encoder is None:
        encoder = build_label_encoder(config["labels"]["classes"])

    train_path = config["data"]["train_file"]
    df_train = read_table(train_path)

    text_col = pick_column(
        df_train,
        config["data"].get(
            "text_column_candidates",
            ["Paragraph", "Sentence", "text", "paragraph"],
        ),
        what="training text column",
    )

    label_col = pick_column(
        df_train,
        config["data"].get(
            "train_label_column_candidates",
            ["Rating", "True Label", "Label", "label"],
        ),
        what="training label column",
    )

    df_train = df_train[[text_col, label_col]].copy()
    df_train = clean_text_column(df_train, text_col)
    df_train = map_train_labels(df_train, label_col=label_col, encoder=encoder)

    return df_train[["text", "labels", "label_text"]].reset_index(drop=True)


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer,
    text_col: str = "text",
    label_col: str = "labels",
):
    """Convert dataframe to a tokenized Hugging Face Dataset."""
    required = {text_col, label_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for tokenization: {sorted(missing)}")

    dataset = Dataset.from_pandas(df[[text_col, label_col]].copy(), preserve_index=False)

    def tok(batch):
        return tokenizer(batch[text_col], padding=True, truncation=True)

    return dataset.map(tok, batched=True)


def dataframe_to_texts_labels(df: pd.DataFrame):
    """Return texts and integer labels from a mapped dataframe."""
    if "text" not in df.columns or "labels" not in df.columns:
        raise KeyError("Expected dataframe to contain 'text' and 'labels' columns.")

    texts = df["text"].astype(str).tolist()
    labels = df["labels"].astype(int).values

    return texts, labels
