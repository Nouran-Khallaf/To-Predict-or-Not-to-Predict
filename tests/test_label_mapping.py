from uncertainty_benchmark.data.label_mapping import (
    to_binary_from_true_label,
    to_binary_train_label,
)


def test_eval_label_mapping():
    assert to_binary_from_true_label(0) == "simple"
    assert to_binary_from_true_label(2) == "simple"
    assert to_binary_from_true_label(3) == "simple"
    assert to_binary_from_true_label(1) == "complex"
    assert to_binary_from_true_label(5) == "complex"


def test_train_label_mapping():
    assert to_binary_train_label(2) == "simple"
    assert to_binary_train_label(3) == "simple"
    assert to_binary_train_label(5) == "complex"
