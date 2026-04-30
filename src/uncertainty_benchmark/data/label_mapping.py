"""Dataset-specific label mapping utilities.

The current default mapping follows the English readability setup.

Evaluation prediction files:
    0, 2, 3 -> simple
    1, 5    -> complex

Training files:
    1, 2, 3 -> simple
    5       -> complex
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder


DEFAULT_CLASSES = ["simple", "complex"]


def build_label_encoder(classes=None) -> LabelEncoder:
    """Create a fixed-order label encoder."""
    if classes is None:
        classes = DEFAULT_CLASSES

    encoder = LabelEncoder()
    encoder.fit(list(classes))
    return encoder


def to_binary_from_true_label(value):
    """Map raw evaluation labels to simple/complex."""
    s = str(value).strip().lower()

    try:
        i = int(float(s))
        if i in (0, 2, 3):
            return "simple"
        if i in (1, 5):
            return "complex"
        return None
    except Exception:
        if s in {"simple", "complex"}:
            return s
        return None


def to_binary_train_label(value):
    """Map raw training labels to simple/complex."""
    s = str(value).strip().lower()

    try:
        i = int(float(s))
        if i in (1, 2, 3):
            return "simple"
        if i == 5:
            return "complex"
        return None
    except Exception:
        if s in {"simple", "complex"}:
            return s
        return None


def map_eval_labels(
    df: pd.DataFrame,
    label_col: str,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Add mapped text labels and integer labels to eval dataframe."""
    out = df.copy()
    out["label_text"] = out[label_col].apply(to_binary_from_true_label)
    out = out.dropna(subset=["label_text"]).copy()
    out["labels"] = encoder.transform(out["label_text"])
    return out


def map_train_labels(
    df: pd.DataFrame,
    label_col: str,
    encoder: LabelEncoder,
) -> pd.DataFrame:
    """Add mapped text labels and integer labels to train dataframe."""
    out = df.copy()
    out["label_text"] = out[label_col].apply(to_binary_train_label)
    out = out.dropna(subset=["label_text"]).copy()
    out["labels"] = encoder.transform(out["label_text"])
    return out
