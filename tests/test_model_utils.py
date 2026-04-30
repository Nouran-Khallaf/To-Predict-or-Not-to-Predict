import numpy as np
import torch

from uncertainty_benchmark.models.dropout import (
    DropoutMC,
    activate_mc_dropout,
    convert_dropouts,
    count_mc_dropout_layers,
)
from uncertainty_benchmark.models.predictors import majority_vote_predictions


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


def test_convert_dropouts_replaces_dropout():
    model = TinyModel()

    assert count_mc_dropout_layers(model) == 0

    convert_dropouts(model, inference_prob=0.1)

    assert count_mc_dropout_layers(model) == 1
    assert any(isinstance(layer, DropoutMC) for layer in model.modules())


def test_activate_mc_dropout_changes_flag():
    model = TinyModel()
    convert_dropouts(model, inference_prob=0.1)

    activate_mc_dropout(model, activate=True)

    layers = [layer for layer in model.modules() if isinstance(layer, DropoutMC)]
    assert len(layers) == 1
    assert layers[0].activate is True

    activate_mc_dropout(model, activate=False)
    assert layers[0].activate is False


def test_mc_dropout_stochastic_in_eval_mode():
    torch.manual_seed(42)

    model = TinyModel()
    convert_dropouts(model, inference_prob=0.9)
    model.eval()
    activate_mc_dropout(model, activate=True)

    x = torch.ones(8, 4)
    y1 = model(x)
    y2 = model(x)

    assert not torch.allclose(y1, y2)


def test_majority_vote_predictions():
    sampled = np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 1, 1, 1],
        ]
    )

    voted = majority_vote_predictions(sampled)

    assert np.array_equal(voted, np.array([0, 1, 1]))
