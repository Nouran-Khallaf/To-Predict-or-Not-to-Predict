import pytest

from uncertainty_benchmark.config import (
    load_config,
    normalise_config,
    resolve_fold_ids_from_config,
    validate_config,
)


def minimal_config():
    return {
        "experiment_name": "test_exp",
        "model": {
            "model_id_template": "dummy-model-{fold_id}",
        },
        "data": {
            "train_file": "./data/raw/train.xlsx",
            "pred_csv_template": "./data/predictions/pred_fold_{fold_id}.csv",
            "lang_name": "test_lang",
        },
        "methods": {
            "enabled": ["SR", "ENT"],
        },
    }


def test_normalise_config_adds_defaults():
    cfg = normalise_config(minimal_config())

    assert cfg["folds"]["n_folds"] == 1
    assert cfg["mc_dropout"]["committee_size"] == 20
    assert cfg["metrics"]["ece_bins"] == 15
    assert cfg["outputs"]["outdir"]


def test_resolve_fold_ids_explicit():
    cfg = normalise_config(
        {
            **minimal_config(),
            "folds": {"fold_ids": [0, 2, 5]},
        }
    )

    assert resolve_fold_ids_from_config(cfg) == [0, 2, 5]


def test_resolve_fold_ids_from_n_folds():
    cfg = normalise_config(
        {
            **minimal_config(),
            "folds": {"n_folds": 3, "fold_ids": None},
        }
    )

    assert resolve_fold_ids_from_config(cfg) == [0, 1, 2]


def test_validate_config_accepts_minimal():
    cfg = normalise_config(minimal_config())
    validate_config(cfg)


def test_validate_config_rejects_unknown_method():
    cfg = normalise_config(
        {
            **minimal_config(),
            "methods": {"enabled": ["SR", "BAD_METHOD"]},
        }
    )

    with pytest.raises(ValueError):
        validate_config(cfg)


def test_validate_config_rejects_bad_dropout():
    cfg = normalise_config(
        {
            **minimal_config(),
            "mc_dropout": {"dropout_p": 1.5},
        }
    )

    with pytest.raises(ValueError):
        validate_config(cfg)


def test_load_example_config():
    cfg = load_config("configs/example_config.yaml")
    assert "experiment_name" in cfg
    assert "methods" in cfg
    assert "outputs" in cfg
