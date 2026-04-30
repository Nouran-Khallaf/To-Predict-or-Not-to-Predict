"""Data utilities."""

from uncertainty_benchmark.data.label_mapping import (
    DEFAULT_CLASSES,
    build_label_encoder,
    map_eval_labels,
    map_train_labels,
    to_binary_from_true_label,
    to_binary_train_label,
)
from uncertainty_benchmark.data.loaders import (
    clean_text_column,
    dataframe_to_texts_labels,
    load_eval_from_prediction_csv,
    load_train_file,
    pick_column,
    read_table,
    tokenize_dataframe,
)
from uncertainty_benchmark.data.overlap import (
    normalise_text_series,
    remove_predicted_rows_by_file,
    remove_text_overlap,
)

__all__ = [
    "DEFAULT_CLASSES",
    "build_label_encoder",
    "map_eval_labels",
    "map_train_labels",
    "to_binary_from_true_label",
    "to_binary_train_label",
    "clean_text_column",
    "dataframe_to_texts_labels",
    "load_eval_from_prediction_csv",
    "load_train_file",
    "pick_column",
    "read_table",
    "tokenize_dataframe",
    "normalise_text_series",
    "remove_predicted_rows_by_file",
    "remove_text_overlap",
]
