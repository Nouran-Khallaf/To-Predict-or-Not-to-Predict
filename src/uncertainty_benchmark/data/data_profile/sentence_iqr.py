#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uncertainty_benchmark.data_profile.sentence_iqr

Reusable utilities for profiling sentence-token-length IQRs by label.

This module supports the script:

    scripts/profile_sentence_iqr.py

It can:
- read TSV/TXT/CSV/Excel files
- auto-detect text and label columns
- map labels to binary simple/complex labels
- split texts into sentences using spaCy
- compute token-length IQRs by label
- build sentence-level and example-level summary DataFrames

Main convention
---------------
Binary labels are assumed to be:

    0 = simple
    1 = complex

The special `rating_cefr235` mode maps:

    2 -> simple / 0
    3 -> simple / 0
    5 -> complex / 1
    4 -> excluded
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Column detection configuration
# ---------------------------------------------------------------------

CANDIDATE_ID_COLUMNS = ["id", "index"]
CANDIDATE_NAME_COLUMNS = ["name", "domain", "sub-domain", "subdomain", "context"]
CANDIDATE_TEXT_COLUMNS = ["sentence", "paragraph", "text"]
CANDIDATE_LABEL_COLUMNS = ["label", "rating", "class", "true label", "target", "y"]

WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


# ---------------------------------------------------------------------
# Text and column helpers
# ---------------------------------------------------------------------

def normalise_column_name(name: object) -> str:
    """Normalise a column name for matching."""
    return re.sub(r"\s+", " ", str(name).replace("\u00A0", " ")).strip().lower()


def clean_text(value: object) -> str:
    """Clean raw text before sentence splitting."""
    if not isinstance(value, str):
        return ""
    value = value.replace("\xad", " ")  # soft hyphen
    return WHITESPACE_RE.sub(" ", value).strip()


def pick_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    override: Optional[str] = None,
) -> Optional[str]:
    """Pick a column using an override or candidate names."""
    norm_map = {normalise_column_name(col): col for col in df.columns}

    if override:
        key = normalise_column_name(override)
        if key in norm_map:
            return norm_map[key]
        raise ValueError(
            f"Override column {override!r} was not found. Available columns: {list(df.columns)}"
        )

    for candidate in candidates:
        key = normalise_column_name(candidate)
        if key in norm_map:
            return norm_map[key]

    return None


def resolve_columns(
    df: pd.DataFrame,
    override_id: Optional[str] = None,
    override_name: Optional[str] = None,
    override_text: Optional[str] = None,
    override_label: Optional[str] = None,
) -> Tuple[str, str, str, str]:
    """Resolve ID, name/domain, text, and label columns.

    ID and name columns are optional and are synthesised when missing.
    Text and label columns are required.

    Returns
    -------
    id_col, name_col, text_col, label_col
    """
    id_col = pick_column(df, CANDIDATE_ID_COLUMNS, override_id)
    name_col = pick_column(df, CANDIDATE_NAME_COLUMNS, override_name)
    text_col = pick_column(df, CANDIDATE_TEXT_COLUMNS, override_text)
    label_col = pick_column(df, CANDIDATE_LABEL_COLUMNS, override_label)

    if text_col is None:
        raise ValueError(
            f"Could not find a text column. Looked for {CANDIDATE_TEXT_COLUMNS}. "
            f"Available columns: {list(df.columns)}"
        )

    if label_col is None:
        raise ValueError(
            f"Could not find a label column. Looked for {CANDIDATE_LABEL_COLUMNS}. "
            f"Available columns: {list(df.columns)}"
        )

    if id_col is None:
        id_col = "__row_index__"
        df[id_col] = range(len(df))

    if name_col is None:
        name_col = "__name__"

    return id_col, name_col, text_col, label_col


# ---------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------

def read_input_file(path: str | Path, sheet: Optional[Union[str, int]] = None) -> pd.DataFrame:
    """Read TSV/TXT/CSV/Excel file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".tsv", ".txt", ".ol"}:
        return pd.read_csv(path, sep="\t", dtype=str, engine="python")

    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, engine="python")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, dtype=str)

    raise ValueError(f"Unsupported file type: {path}")


def collect_files(inputs: Sequence[str | Path], patterns: Sequence[str]) -> List[Path]:
    """Collect files from explicit file paths and directories."""
    files: List[Path] = []

    for item in inputs:
        path = Path(item)

        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for pattern in patterns:
                files.extend(sorted(path.rglob(pattern)))
        else:
            print(f"[warn] Skipping non-existent path: {item}", file=sys.stderr)

    seen = set()
    unique: List[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    return unique


# ---------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------

def choose_effective_label_mode(label_col: str, requested_mode: str = "auto") -> str:
    """Choose label mode, resolving 'auto'."""
    if requested_mode != "auto":
        return requested_mode

    if normalise_column_name(label_col) == "rating":
        return "rating_cefr235"

    return "generic"


def map_labels_to_binary(labels: pd.Series, mode: str = "generic") -> pd.Series:
    """Map labels to binary values 0=simple and 1=complex.

    Returns a Series that may contain NaN for excluded/unmapped labels.
    """
    if mode == "rating_cefr235":
        def to_int(value: object) -> Optional[int]:
            try:
                return int(float(str(value).strip()))
            except Exception:
                return None

        numeric = labels.map(to_int)
        keep = numeric.isin({2, 3, 5})
        numeric = numeric.where(keep)
        return numeric.map({2: 0, 3: 0, 5: 1})

    if mode != "generic":
        raise ValueError(f"Unknown label mode: {mode}")

    def to_binary(value: object) -> int:
        text = str(value).strip().lower()

        if text in {"0", "simple", "simp", "easy", "neg", "negative", "false", "no"}:
            return 0
        if text in {"1", "complex", "comp", "hard", "pos", "positive", "true", "yes"}:
            return 1

        try:
            numeric = int(float(text))
            if numeric in {0, 1}:
                return numeric
        except Exception:
            pass

        raise ValueError(f"Unrecognised label value: {value!r}")

    return labels.map(to_binary)


# ---------------------------------------------------------------------
# spaCy sentence splitting
# ---------------------------------------------------------------------

def load_spacy_model(lang: str, use_parser: bool = False):
    """Load spaCy model, falling back to a blank sentencizer pipeline."""
    import spacy

    model_map = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "it": "it_core_news_sm",
        "es": "es_core_news_sm",
        "ca": "ca_core_news_sm",
        "ru": "ru_core_news_sm",
        "ar": "ar_core_news_sm",
        "hi": "xx_sent_ud_sm",
    }

    model_name = model_map.get(lang)
    nlp = None

    if model_name:
        try:
            if use_parser:
                nlp = spacy.load(model_name)
            else:
                nlp = spacy.load(
                    model_name,
                    disable=["tagger", "ner", "lemmatizer", "morphologizer"],
                )
        except Exception:
            nlp = None

    if nlp is None:
        try:
            nlp = spacy.blank(lang)
        except Exception:
            nlp = spacy.blank("xx")

    if use_parser:
        if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    else:
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    return nlp


def sentence_spans(text: str, nlp) -> List[object]:
    """Return non-empty spaCy sentence spans."""
    if not text:
        return []

    doc = nlp(text)
    return [sent for sent in doc.sents if sent.text.strip()]


def token_length(sentence_span) -> int:
    """Count non-space tokens in a sentence span."""
    return sum(1 for token in sentence_span if not token.is_space)


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def process_dataframe(
    df: pd.DataFrame,
    source_name: str,
    nlp,
    col_overrides: Optional[Dict[str, Optional[str]]] = None,
    label_mode: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process one DataFrame into sentence-level and example-level rows.

    Parameters
    ----------
    df:
        Input DataFrame.
    source_name:
        Name stored in the output `file` column.
    nlp:
        spaCy pipeline.
    col_overrides:
        Optional keys: id, name, text, label.
    label_mode:
        'auto', 'generic', or 'rating_cefr235'.

    Returns
    -------
    sentence_df, example_df
    """
    col_overrides = col_overrides or {}
    df = df.copy()

    id_col, name_col, text_col, label_col = resolve_columns(
        df,
        override_id=col_overrides.get("id"),
        override_name=col_overrides.get("name"),
        override_text=col_overrides.get("text"),
        override_label=col_overrides.get("label"),
    )

    if name_col == "__name__":
        df[name_col] = Path(source_name).stem

    df["__clean_text__"] = df[text_col].map(clean_text)

    effective_label_mode = choose_effective_label_mode(label_col, label_mode)
    labels = map_labels_to_binary(df[label_col], mode=effective_label_mode)

    keep = labels.notna()
    df = df.loc[keep].copy()
    labels = labels.loc[keep].astype(int)

    sentence_rows: List[Dict[str, object]] = []
    example_rows: List[Dict[str, object]] = []

    for row_idx, row in df.iterrows():
        label = int(labels.loc[row_idx])

        example_rows.append(
            {
                "label": label,
                "file": source_name,
                "id": row[id_col],
                "name": row[name_col],
            }
        )

        for sent_idx, sent in enumerate(sentence_spans(row["__clean_text__"], nlp)):
            sentence_rows.append(
                {
                    "label": label,
                    "sent_len": token_length(sent),
                    "sentence": sent.text.strip(),
                    "sentence_index": sent_idx,
                    "file": source_name,
                    "id": row[id_col],
                    "name": row[name_col],
                }
            )

    return pd.DataFrame(sentence_rows), pd.DataFrame(example_rows)


def process_files(
    files: Sequence[str | Path],
    lang_code: str,
    use_parser: bool = False,
    sheet: Optional[Union[str, int]] = None,
    col_overrides: Optional[Dict[str, Optional[str]]] = None,
    label_mode: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process multiple files and return combined sentence/example DataFrames."""
    nlp = load_spacy_model(lang_code, use_parser=use_parser)
    all_sentence_dfs: List[pd.DataFrame] = []
    all_example_dfs: List[pd.DataFrame] = []

    for file_path in files:
        path = Path(file_path)
        df = read_input_file(path, sheet=sheet)
        sentence_df, example_df = process_dataframe(
            df=df,
            source_name=path.name,
            nlp=nlp,
            col_overrides=col_overrides,
            label_mode=label_mode,
        )
        all_sentence_dfs.append(sentence_df)
        all_example_dfs.append(example_df)

    if all_sentence_dfs:
        sentences = pd.concat(all_sentence_dfs, ignore_index=True)
    else:
        sentences = pd.DataFrame(columns=["label", "sent_len", "sentence", "sentence_index", "file", "id", "name"])

    if all_example_dfs:
        examples = pd.concat(all_example_dfs, ignore_index=True)
    else:
        examples = pd.DataFrame(columns=["label", "file", "id", "name"])

    return sentences, examples


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def iqr_range(values: Iterable[float]) -> Tuple[int, Optional[int], Optional[int]]:
    """Return n, rounded Q1, and rounded Q3."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return 0, None, None

    q1 = float(np.percentile(series, 25))
    q3 = float(np.percentile(series, 75))
    return int(series.shape[0]), int(round(q1)), int(round(q3))


def sentence_summary_by_label(sentence_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise sentence-token lengths by label."""
    rows: List[Dict[str, object]] = []

    if sentence_df.empty:
        return pd.DataFrame(columns=["label", "sentence_count", "q1", "q3", "median", "mean", "std"])

    for label, sub in sentence_df.groupby("label"):
        lengths = pd.to_numeric(sub["sent_len"], errors="coerce").dropna()
        n, q1, q3 = iqr_range(lengths)
        rows.append(
            {
                "label": int(label),
                "sentence_count": n,
                "q1": q1,
                "q3": q3,
                "median": float(np.median(lengths)) if not lengths.empty else np.nan,
                "mean": float(np.mean(lengths)) if not lengths.empty else np.nan,
                "std": float(np.std(lengths)) if not lengths.empty else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("label")


def example_summary_by_label(example_df: pd.DataFrame) -> pd.DataFrame:
    """Count original examples by label."""
    if example_df.empty:
        return pd.DataFrame(columns=["label", "example_count"])

    counts = example_df["label"].value_counts().sort_index()
    return pd.DataFrame(
        {
            "label": counts.index.astype(int),
            "example_count": counts.values.astype(int),
        }
    )


def sentence_iqr_for_labels(
    sentence_df: pd.DataFrame,
    simple_label: int = 0,
    complex_label: int = 1,
) -> Dict[str, object]:
    """Compute simple/complex sentence-length IQR values."""
    simple = sentence_df[sentence_df["label"] == simple_label]
    complex_ = sentence_df[sentence_df["label"] == complex_label]

    n_simple, q1_simple, q3_simple = iqr_range(simple["sent_len"] if not simple.empty else [])
    n_complex, q1_complex, q3_complex = iqr_range(complex_["sent_len"] if not complex_.empty else [])

    return {
        "simple_label": simple_label,
        "complex_label": complex_label,
        "simple_sentence_count": n_simple,
        "simple_q1": q1_simple,
        "simple_q3": q3_simple,
        "complex_sentence_count": n_complex,
        "complex_q1": q1_complex,
        "complex_q3": q3_complex,
    }


def profile_sentence_iqr(
    inputs: Sequence[str | Path],
    lang_code: str,
    patterns: Sequence[str] = ("*.tsv", "*.txt", "*.ol", "*.xlsx", "*.xls", "*.csv"),
    use_parser: bool = False,
    sheet: Optional[Union[str, int]] = None,
    col_overrides: Optional[Dict[str, Optional[str]]] = None,
    label_mode: str = "auto",
    simple_label: int = 0,
    complex_label: int = 1,
) -> Dict[str, object]:
    """High-level helper for profiling sentence-token-length IQRs.

    Returns a dictionary containing:
      - files
      - sentences
      - examples
      - sentence_summary
      - example_summary
      - iqr
    """
    files = collect_files(inputs, patterns)
    if not files:
        raise FileNotFoundError("No input files found.")

    sentences, examples = process_files(
        files=files,
        lang_code=lang_code,
        use_parser=use_parser,
        sheet=sheet,
        col_overrides=col_overrides,
        label_mode=label_mode,
    )

    sentence_summary = sentence_summary_by_label(sentences)
    example_summary = example_summary_by_label(examples)
    iqr = sentence_iqr_for_labels(sentences, simple_label=simple_label, complex_label=complex_label)

    return {
        "files": files,
        "sentences": sentences,
        "examples": examples,
        "sentence_summary": sentence_summary,
        "example_summary": example_summary,
        "iqr": iqr,
    }


# ---------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------

def save_profile_outputs(
    outdir: str | Path,
    sentences: pd.DataFrame,
    examples: pd.DataFrame,
    sentence_summary: Optional[pd.DataFrame] = None,
    example_summary: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """Save sentence-level and summary CSVs.

    Returns output paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if sentence_summary is None:
        sentence_summary = sentence_summary_by_label(sentences)
    if example_summary is None:
        example_summary = example_summary_by_label(examples)

    paths = {
        "sentences": outdir / "sentences_token_lengths.csv",
        "sentence_summary": outdir / "label_summaries_sentences.csv",
        "example_summary": outdir / "example_label_counts.csv",
    }

    sentences.to_csv(paths["sentences"], index=False, encoding="utf-8")
    sentence_summary.to_csv(paths["sentence_summary"], index=False, encoding="utf-8")
    example_summary.to_csv(paths["example_summary"], index=False, encoding="utf-8")

    return paths


__all__ = [
    "CANDIDATE_ID_COLUMNS",
    "CANDIDATE_NAME_COLUMNS",
    "CANDIDATE_TEXT_COLUMNS",
    "CANDIDATE_LABEL_COLUMNS",
    "normalise_column_name",
    "clean_text",
    "pick_column",
    "resolve_columns",
    "read_input_file",
    "collect_files",
    "choose_effective_label_mode",
    "map_labels_to_binary",
    "load_spacy_model",
    "sentence_spans",
    "token_length",
    "process_dataframe",
    "process_files",
    "iqr_range",
    "sentence_summary_by_label",
    "example_summary_by_label",
    "sentence_iqr_for_labels",
    "profile_sentence_iqr",
    "save_profile_outputs",
]
