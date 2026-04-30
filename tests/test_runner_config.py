from uncertainty_benchmark.runner import (
    get_enabled_methods,
    get_required_methods,
    resolve_fold_ids,
)


def test_resolve_fold_ids_explicit():
    config = {"folds": {"fold_ids": [0, 3, 7]}}
    assert resolve_fold_ids(config) == [0, 3, 7]


def test_resolve_fold_ids_from_n_folds():
    config = {"folds": {"n_folds": 3}}
    assert resolve_fold_ids(config) == [0, 1, 2]


def test_enabled_methods_stable_order():
    config = {"methods": {"enabled": ["BALD", "SR", "ENT"]}}
    assert get_enabled_methods(config) == ["SR", "ENT", "BALD"]


def test_required_methods_adds_huq_dependencies():
    enabled = ["HUQ-MD"]
    required = get_required_methods(enabled)

    assert "HUQ-MD" in required
    assert "SR" in required
    assert "MD" in required
