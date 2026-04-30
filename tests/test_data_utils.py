from pathlib import Path

import pandas as pd

from uncertainty_benchmark.data.label_mapping import (
    build_label_encoder,
    map_eval_labels,
    map_train_labels,
    to_binary_from_true_label,
    to_binary_train_label,
)
from uncertainty_benchmark.data.loaders import (
    clean_text_column,
    pick_column,
    read_table,
)
from uncertainty_benchmark.data.overlap import remove_text_overlap


def test_eval_label_mapping_values():
    assert to_binary_from_true_label(0) == "simple"
    assert to_binary_from_true_label(2) == "simple"
    assert to_binary_from_true_label(3) == "simple"
    assert to_binary_from_true_label(1) == "complex"
    assert to_binary_from_true_label(5) == "complex"
    assert to_binary_from_true_label("simple") == "simple"
    assert to_binary_from_true_label("complex") == "complex"
    assert to_binary_from_true_label("unknown") is None


def test_train_label_mapping_values():
    assert to_binary_train_label(2) == "simple"
    assert to_binary_train_label(3) == "simple"
    assert to_binary_train_label(5) == "complex"
    assert to_binary_train_label(1) is None


def test_build_label_encoder_order():
    encoder = build_label_encoder(["simple", "complex"])
    assert list(encoder.classes_) == ["complex", "simple"] or set(encoder.classes_) == {
        "simple",
        "complex",
    }

    encoded = encoder.transform(["simple", "complex"])
    decoded = encoder.inverse_transform(encoded)
    assert list(decoded) == ["simple", "complex"]


def test_map_eval_labels():
    encoder = build_label_encoder(["simple", "complex"])
    df = pd.DataFrame({"label": [0, 5, "bad"]})

    out = map_eval_labels(df, label_col="label", encoder=encoder)

    assert len(out) == 2
    assert "labels" in out.columns
    assert "label_text" in out.columns


def test_map_train_labels():
    encoder = build_label_encoder(["simple", "complex"])
    df = pd.DataFrame({"rating": [2, 3, 5, 1]})

    out = map_train_labels(df, label_col="rating", encoder=encoder)

    assert len(out) == 3
    assert set(out["label_text"]) == {"simple", "complex"}


def test_pick_column():
    df = pd.DataFrame({"Sentence": ["a"], "Label": [1]})
    assert pick_column(df, ["text", "Sentence"]) == "Sentence"


def test_clean_text_column():
    df = pd.DataFrame({"Sentence": [" hello ", "", "world"]})
    out = clean_text_column(df, "Sentence")

    assert list(out["text"]) == ["hello", "world"]


def test_remove_text_overlap():
    train = pd.DataFrame({"text": ["a", "b", "c"], "labels": [0, 1, 0]})
    eval_df = pd.DataFrame({"text": ["b"], "labels": [1]})

    out = remove_text_overlap(train, eval_df)

    assert list(out["text"]) == ["a", "c"]


def test_read_table_csv(tmp_path: Path):
    path = tmp_path / "sample.csv"
    path.write_text("Sentence,Label\nhello,1\n", encoding="utf-8")

    df = read_table(path)

    assert list(df.columns) == ["Sentence", "Label"]
    assert len(df) == 1
