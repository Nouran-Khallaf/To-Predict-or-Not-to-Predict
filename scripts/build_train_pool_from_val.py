#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build fold-specific training pool by removing validation instances.

This script uses:
1. The original full dataset files for each language.
2. The saved validation prediction CSV for one fold.

It removes validation sentences from the original files and saves the remaining
rows as the fold-specific training pool.

Example
-------
python scripts/build_train_pool_from_val.py \
  --val-csv data/predictions/bert-base-multilingual-cased_fold_0_val_predictions.csv \
  --original-files \
      readme_en_combined_all=data/original/readme_en_combined_all.xlsx \
      readme_ar_combined_all=data/original/readme_ar_combined_all.xlsx \
      readme_fr_combined_all=data/original/readme_fr_combined_all.xlsx \
      readme_hi_combined_all=data/original/readme_hi_combined_all.xlsx \
      readme_ru_combined_all=data/original/readme_ru_combined_all.xlsx \
  --outdir results/fold_0_train_pool
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_LABEL_THRESHOLD = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train pool by removing validation instances."
    )

    parser.add_argument(
        "--val-csv",
        required=True,
        help="Validation prediction CSV for the fold.",
    )

    parser.add_argument(
        "--original-files",
        nargs="+",
        required=True,
        help=(
            "Language/file pairs in the form lang_key=path. "
            "Example: readme_en_combined_all=data/original/readme_en_combined_all.xlsx"
        ),
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--label-threshold",
        type=int,
        default=DEFAULT_LABEL_THRESHOLD,
        help=(
            "Rating threshold for complex labels. "
            "Default: rating >= 4 is complex; rating < 4 is simple."
        ),
    )

    return parser.parse_args()


def normalise_text(text: object) -> str:
    """Normalise text for robust sentence matching."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_original_files(pairs: list[str]) -> dict[str, Path]:
    """Parse lang_key=path arguments."""
    parsed: dict[str, Path] = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                "Each --original-files item must be in the form lang_key=path. "
                f"Got: {pair}"
            )

        lang_key, path = pair.split("=", 1)
        parsed[lang_key.strip()] = Path(path.strip())

    return parsed


def load_original_file(path: Path, lang_key: str, label_threshold: int) -> pd.DataFrame:
    """Load one original language file and standardise columns."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    required = {"Sentence", "Rating"}
    missing = required.difference(df.columns)

    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Lang"] = lang_key
    df["sentence_norm"] = df["Sentence"].apply(normalise_text)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    df["label"] = df["Rating"].apply(
        lambda x: "complex" if pd.notna(x) and x >= label_threshold else "simple"
    )

    return df


def build_validation_keys(val_df: pd.DataFrame) -> set[tuple[str, str]]:
    """Build validation keys as (Lang, normalised sentence)."""
    required = {"Sentence", "Lang"}
    missing = required.difference(val_df.columns)

    if missing:
        raise KeyError(f"Validation CSV is missing required columns: {sorted(missing)}")

    val_df = val_df.copy()
    val_df["sentence_norm"] = val_df["Sentence"].apply(normalise_text)

    return set(zip(val_df["Lang"].astype(str), val_df["sentence_norm"]))


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    original_files = parse_original_files(args.original_files)

    val_df = pd.read_csv(args.val_csv)
    val_keys = build_validation_keys(val_df)

    all_train_parts = []
    summary_rows = []

    for lang_key, path in original_files.items():
        original_df = load_original_file(
            path=path,
            lang_key=lang_key,
            label_threshold=args.label_threshold,
        )

        original_df["is_validation"] = list(
            zip(original_df["Lang"].astype(str), original_df["sentence_norm"])
        )
        original_df["is_validation"] = original_df["is_validation"].isin(val_keys)

        train_df = original_df[~original_df["is_validation"]].copy()
        removed_df = original_df[original_df["is_validation"]].copy()

        all_train_parts.append(train_df)

        summary_rows.append(
            {
                "Lang": lang_key,
                "original_n": len(original_df),
                "validation_removed_n": len(removed_df),
                "train_pool_n": len(train_df),
                "original_simple_n": int((original_df["label"] == "simple").sum()),
                "original_complex_n": int((original_df["label"] == "complex").sum()),
                "train_simple_n": int((train_df["label"] == "simple").sum()),
                "train_complex_n": int((train_df["label"] == "complex").sum()),
            }
        )

        train_df.to_csv(
            outdir / f"{lang_key}_fold_train_pool.csv",
            index=False,
        )

        removed_df.to_csv(
            outdir / f"{lang_key}_removed_validation_rows.csv",
            index=False,
        )

    combined_train = pd.concat(all_train_parts, ignore_index=True)

    combined_train.to_csv(
        outdir / "combined_fold_train_pool.csv",
        index=False,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        outdir / "train_pool_summary.csv",
        index=False,
    )

    print("Saved train pool files to:", outdir)
    print()
    print(summary_df.to_string(index=False))
    print()
    print("Combined train pool size:", len(combined_train))


if __name__ == "__main__":
    main()