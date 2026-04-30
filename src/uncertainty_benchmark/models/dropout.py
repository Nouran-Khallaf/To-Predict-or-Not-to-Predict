"""MC-dropout conversion and activation utilities."""

from __future__ import annotations

import torch


class DropoutMC(torch.nn.Module):
    """Dropout layer that can be activated during inference."""

    def __init__(self, p: float, activate: bool = False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(
            x,
            self.p,
            training=self.training or self.activate,
        )


def _get_dropout_probability(layer) -> float:
    """Get dropout probability from standard or StableDropout layers."""
    if hasattr(layer, "p"):
        return float(layer.p)

    if hasattr(layer, "drop_prob"):
        return float(layer.drop_prob)

    raise AttributeError(
        f"Could not find dropout probability field on layer {layer.__class__.__name__}"
    )


def convert_to_mc_dropout(model, inference_prob: float | None = None) -> None:
    """Recursively replace Dropout/StableDropout layers with DropoutMC.

    Parameters
    ----------
    model:
        Torch module to modify in place.

    inference_prob:
        If given, use this dropout probability at inference.
        If None, preserve the original layer probability.
    """
    for module_name, child in list(model._modules.items()):
        layer_name = child.__class__.__name__

        if layer_name in {"Dropout", "StableDropout"}:
            original_p = _get_dropout_probability(child)
            p = original_p if inference_prob is None else inference_prob
            model._modules[module_name] = DropoutMC(p=p, activate=False)
        else:
            convert_to_mc_dropout(child, inference_prob=inference_prob)


def convert_dropouts(model, inference_prob: float = 0.1) -> None:
    """Convenience wrapper for MC-dropout conversion."""
    convert_to_mc_dropout(model, inference_prob=inference_prob)


def activate_mc_dropout(model: torch.nn.Module, activate: bool, random_p: float = 0.0) -> None:
    """Activate/deactivate all DropoutMC layers recursively.

    When activated, dropout remains stochastic even if the model is in eval mode.
    """
    for child in model.children():
        if isinstance(child, DropoutMC):
            child.activate = activate

            if activate and random_p:
                child.p = random_p

            if not activate:
                child.p = child.p_init
        else:
            activate_mc_dropout(child, activate=activate, random_p=random_p)


def count_mc_dropout_layers(model: torch.nn.Module) -> int:
    """Count converted DropoutMC layers."""
    return sum(1 for layer in model.modules() if isinstance(layer, DropoutMC))
