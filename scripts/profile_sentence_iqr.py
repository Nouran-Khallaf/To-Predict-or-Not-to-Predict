#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_sentence_iqr.py

Profile sentence-length interquartile ranges (IQRs) by label.

This script reads TSV/TXT/CSV/Excel files, splits each text into sentences using
spaCy, computes token-length IQRs for simple vs complex examples, and prints a
LaTeX table row.

It can also save:
  - sentence-level token lengths
  - per-label sentence summaries
  - original example counts by label

Supported input formats
-----------------------
  - .tsv, .txt, .ol  -> tab-separated
  - .csv             -> comma-separated CSV
  - .xlsx, .xls      -> Excel

Expected columns
----------------
The script auto-detects common column names:
  - text: sentence, paragraph, text
  - label: label, rating, class, true label, target, y
  - optional id: id, index
  - optional name/domain: name, domain, sub-domain, subdomain, context

You can override these using:
  --col-text
  --col-label
  --col-id
  --col-name

Label modes
-----------
1. generic
   Maps labels such as 0/simple/false/no -> 0 and 1/complex/true/yes -> 1.

2. rating_cefr235
   Designed for CEFR-style ratings where:
     - 2 -> simple / A2 -> 0
     - 3 -> simple / B1 -> 0
     - 5 -> complex / C1 -> 1
     - 4 -> dropped / B2 excluded

3. auto
   Uses rating_cefr235 if the detected/provided label column is called Rating;
   otherwise uses generic.

Examples
--------
# TSV files
python scripts/profile_sentence_iqr.py data/es/*.tsv \
  --lang es \
  --language-name "Spanish" \
  --outdir results/data_profile/es

# Excel with explicit columns
python scripts/profile_sentence_iqr.py data/es/*.xlsx \
  --lang es \
  --language-name "Spanish" \
  --sheet Sheet1 \
  --col-text Paragraph \
  --col-label Rating \
  --label-mode rating_cefr235 \
  --outdir results/data_profile/es

# Directory input
python scripts/profile_sentence_iqr.py data/vikidia/es \
  --lang es \
  --language-name "Spanish" \
  --dataset-name "Vikidia" \
  --outdir results/data_profile/vikidia_es
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

CANDIDATE_ID_COLUMNS = ["id", "index"]
CANDIDATE_NAME_COLUMNS = ["name", "domain", "sub-domain", "subdomain", "context"]
CANDIDATE_TEXT_COLUMNS = ["sentence", "paragraph", "text"]
CANDIDATE_LABEL_COLUMNS = ["label", "rating", "class", "true label", "target", "y"]

WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalise_column_name(name: str) -> str:
    """Normalise column names for robust matching."""
    return re.sub(r"\s+", " ", str(name).replace("\u00A0", " ")).strip().lower()


def clean_text(value: object) -> str:
    """Clean input text before sentence splitting."""
    if not isinstance(value, str):
        return ""
    value = value.replace("\xad", " ")  # soft hyphen
    return WHITESPACE_RE.sub(" ", value).strip()


def pick_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    override: Optional[str] = None,
) -> Optional[str]:
    """Pick a column by override or candidate names."""
    norm_map = {normalise_column_name(c): c for c in df.columns}

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
    """Resolve id, name, text, and label columns.

    ID and name columns are optional and are synthesised if missing.
    Text and label columns are required.
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
# Loading files
# ---------------------------------------------------------------------

def read_input_file(path: Path, sheet: Optional[Union[str, int]] = None) -> pd.DataFrame:
    """Read a supported input file into a DataFrame."""
    suffix = path.suffix.lower()

    if suffix in {".tsv", ".txt", ".ol"}:
        return pd.read_csv(path, sep="\t", dtype=str, engine="python")

    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, engine="python")

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, dtype=str)

    raise ValueError(f"Unsupported file type: {path}")


def collect_files(inputs: Sequence[str], patterns: Sequence[str]) -> List[Path]:
    """Collect input files from paths and directories."""
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

    # De-duplicate while preserving order.
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

def map_labels_to_binary(labels: pd.Series, mode: str) -> pd.Series:
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
        dropped = int((~keep).sum())
        if dropped:
            print(f"[info] Dropping {dropped} row(s) with rating not in {{2, 3, 5}}.")

        numeric = numeric.where(keep)
        return numeric.map({2: 0, 3: 0, 5: 1})

    if mode != "generic":
        raise ValueError(f"Unknown label mode: {mode}")

    def to_binary(value: object) -> int:
        s = str(value).strip().lower()

        if s in {"0", "simple", "simp", "easy", "neg", "negative", "false", "no"}:
            return 0
        if s in {"1", "complex", "comp", "hard", "pos", "positive", "true", "yes"}:
            return 1

        try:
            numeric = int(float(s))
            if numeric in {0, 1}:
                return numeric
        except Exception:
            pass

        raise ValueError(f"Unrecognised label value: {value!r}")

    return labels.map(to_binary)


def choose_effective_label_mode(label_col: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    if normalise_column_name(label_col) == "rating":
        return "rating_cefr235"

    return "generic"


# ---------------------------------------------------------------------
# spaCy sentence splitting
# ---------------------------------------------------------------------

def load_spacy_model(lang: str, use_parser: bool = False):
    """Load a spaCy model, falling back to a blank pipeline with sentencizer."""
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
    """Return non-empty sentence spans from text."""
    if not text:
        return []

    doc = nlp(text)
    return [sent for sent in doc.sents if sent.text.strip()]


def token_length(sent) -> int:
    """Count non-space tokens in a sentence span."""
    return sum(1 for token in sent if not token.is_space)


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------

def process_files(
    files: Sequence[Path],
    lang_code: str,
    use_parser: bool,
    sheet: Optional[Union[str, int]],
    col_overrides: Dict[str, Optional[str]],
    label_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process input files and return sentence-level and example-level data.

    Returns
    -------
    sentence_df:
        One row per sentence, with label and token length.
    example_df:
        One row per original example after label filtering.
    """
    nlp = load_spacy_model(lang_code, use_parser=use_parser)

    sentence_rows: List[Dict[str, object]] = []
    example_rows: List[Dict[str, object]] = []

    for path in files:
        df = read_input_file(path, sheet=sheet)

        id_col, name_col, text_col, label_col = resolve_columns(
            df,
            override_id=col_overrides.get("id"),
            override_name=col_overrides.get("name"),
            override_text=col_overrides.get("text"),
            override_label=col_overrides.get("label"),
        )

        if name_col == "__name__":
            df[name_col] = path.stem

        df["__clean_text__"] = df[text_col].map(clean_text)

        effective_mode = choose_effective_label_mode(label_col, label_mode)
        labels = map_labels_to_binary(df[label_col], mode=effective_mode)

        keep = labels.notna()
        dropped = int((~keep).sum())
        if dropped:
            print(f"[info] Filtering out {dropped} row(s) in {path.name} due to invalid/excluded labels.")

        df = df.loc[keep].copy()
        labels = labels.loc[keep].astype(int)

        for _, row in df.iterrows():
            label = int(labels.loc[row.name])
            example_rows.append(
                {
                    "label": label,
                    "file": path.name,
                    "id": row[id_col],
                    "name": row[name_col],
                }
            )

            text = row["__clean_text__"]
            for sent_idx, sent in enumerate(sentence_spans(text, nlp)):
                sentence_rows.append(
                    {
                        "label": label,
                        "sent_len": token_length(sent),
                        "sentence": sent.text.strip(),
                        "sentence_index": sent_idx,
                        "file": path.name,
                        "id": row[id_col],
                        "name": row[name_col],
                    }
                )

    return pd.DataFrame(sentence_rows), pd.DataFrame(example_rows)


# ---------------------------------------------------------------------
# Summaries and LaTeX
# ---------------------------------------------------------------------

def iqr_range(values: pd.Series) -> Tuple[int, Optional[int], Optional[int]]:
    """Return n, rounded Q1, and rounded Q3."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 0, None, None

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    return int(values.shape[0]), int(round(q1)), int(round(q3))


def make_sentence_summary(sentence_df: pd.DataFrame) -> pd.DataFrame:
    """Create per-label sentence-length summary."""
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


def make_example_summary(example_df: pd.DataFrame) -> pd.DataFrame:
    """Create per-label original-example count summary."""
    if example_df.empty:
        return pd.DataFrame(columns=["label", "example_count"])

    counts = example_df["label"].value_counts().sort_index()
    return pd.DataFrame(
        {
            "label": counts.index.astype(int),
            "example_count": counts.values.astype(int),
        }
    )


def latex_iqr(q1: Optional[int], q3: Optional[int]) -> str:
    if q1 is None or q3 is None:
        return ""
    return f"{q1}--{q3}"


def latex_escape(text: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def make_latex_row(
    sentence_df: pd.DataFrame,
    language_name: str,
    simple_label: int,
    complex_label: int,
) -> str:
    """Create a LaTeX table row using sentence-level IQRs."""
    simple = sentence_df[sentence_df["label"] == simple_label]
    complex_ = sentence_df[sentence_df["label"] == complex_label]

    n_simple, q1_simple, q3_simple = iqr_range(simple["sent_len"] if not simple.empty else pd.Series(dtype=float))
    n_complex, q1_complex, q3_complex = iqr_range(complex_["sent_len"] if not complex_.empty else pd.Series(dtype=float))

    return (
        f"{latex_escape(language_name)} & "
        f"{n_simple} & {latex_iqr(q1_simple, q3_simple)} & "
        f"{n_complex} & {latex_iqr(q1_complex, q3_complex)} \\\\"
    )


def save_outputs(
    outdir: Path,
    sentence_df: pd.DataFrame,
    example_df: pd.DataFrame,
    language_name: str,
    dataset_name: Optional[str],
    simple_label: int,
    complex_label: int,
) -> None:
    """Save CSV summaries and a small text file with the LaTeX row."""
    outdir.mkdir(parents=True, exist_ok=True)

    sentence_df.to_csv(outdir / "sentences_token_lengths.csv", index=False, encoding="utf-8")

    sentence_summary = make_sentence_summary(sentence_df)
    sentence_summary.to_csv(outdir / "label_summaries_sentences.csv", index=False, encoding="utf-8")

    example_summary = make_example_summary(example_df)
    example_summary.to_csv(outdir / "example_label_counts.csv", index=False, encoding="utf-8")

    lines: List[str] = []
    if dataset_name:
        lines.append(f"\\textbf{{{latex_escape(dataset_name)}}} &  &  &  &  \\\\")
    lines.append(make_latex_row(sentence_df, language_name, simple_label, complex_label))

    (outdir / "latex_iqr_row.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_sheet(value: Optional[str]) -> Optional[Union[str, int]]:
    if value is None:
        return None
    if str(value).isdigit():
        return int(value)
    return value


def parse_patterns(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sentence-token-length IQRs by label and print a LaTeX row."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories. Supported: TSV, TXT, CSV, XLSX, XLS.",
    )
    parser.add_argument(
        "--lang",
        required=True,
        help="Language code for spaCy, e.g. es, en, fr, ru, ar, hi.",
    )
    parser.add_argument(
        "--language-name",
        required=True,
        help="Language name to print in the LaTeX row, e.g. Spanish.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset heading row for LaTeX output, e.g. Vikidia.",
    )
    parser.add_argument(
        "--simple-label",
        type=int,
        default=0,
        help="Numeric label for simple examples. Default: 0.",
    )
    parser.add_argument(
        "--complex-label",
        type=int,
        default=1,
        help="Numeric label for complex examples. Default: 1.",
    )
    parser.add_argument(
        "--use-parser",
        action="store_true",
        help="Use the dependency parser for sentence boundaries when available. Slower but sometimes more accurate.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name or index. Default: first sheet.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional output directory for CSV summaries and LaTeX row.",
    )
    parser.add_argument(
        "--glob",
        default="*.tsv,*.txt,*.ol,*.xlsx,*.xls,*.csv",
        help="Comma-separated file patterns used when an input is a directory.",
    )

    parser.add_argument("--col-id", default=None, help="Override ID column.")
    parser.add_argument("--col-name", default=None, help="Override name/domain column.")
    parser.add_argument("--col-text", default=None, help="Override text column, e.g. Sentence or Paragraph.")
    parser.add_argument("--col-label", default=None, help="Override label column, e.g. Label or Rating.")

    parser.add_argument(
        "--label-mode",
        default="auto",
        choices=["auto", "generic", "rating_cefr235"],
        help="How to map labels to 0/1. Default: auto.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sheet = parse_sheet(args.sheet)
    patterns = parse_patterns(args.glob)
    files = collect_files(args.inputs, patterns)

    if not files:
        print("[error] No input files found.", file=sys.stderr)
        return 1

    print(f"[info] Found {len(files)} input file(s).")
    for path in files:
        print(f"       - {path}")

    col_overrides = {
        "id": args.col_id,
        "name": args.col_name,
        "text": args.col_text,
        "label": args.col_label,
    }

    sentence_df, example_df = process_files(
        files=files,
        lang_code=args.lang,
        use_parser=args.use_parser,
        sheet=sheet,
        col_overrides=col_overrides,
        label_mode=args.label_mode,
    )

    simple_examples = int((example_df["label"] == args.simple_label).sum()) if not example_df.empty else 0
    complex_examples = int((example_df["label"] == args.complex_label).sum()) if not example_df.empty else 0

    print(f"#Examples (rows) — Simple={simple_examples}, Complex={complex_examples}")

    if args.dataset_name:
        print(f"\\textbf{{{latex_escape(args.dataset_name)}}} &  &  &  &  \\\\")

    latex_row = make_latex_row(
        sentence_df,
        language_name=args.language_name,
        simple_label=args.simple_label,
        complex_label=args.complex_label,
    )
    print(latex_row)

    if args.outdir:
        outdir = Path(args.outdir)
        save_outputs(
            outdir=outdir,
            sentence_df=sentence_df,
            example_df=example_df,
            language_name=args.language_name,
            dataset_name=args.dataset_name,
            simple_label=args.simple_label,
            complex_label=args.complex_label,
        )
        print(f"[ok] Wrote outputs to: {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
