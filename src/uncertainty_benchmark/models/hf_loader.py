"""Hugging Face model/tokenizer loading."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_device(device: str | None = None) -> torch.device:
    """Return the requested device or choose CUDA when available."""
    if device is not None:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    model_id: str,
    device: str | None = None,
    trust_remote_code: bool = False,
    **model_kwargs,
):
    """Load a sequence-classification model and tokenizer.

    Parameters
    ----------
    model_id:
        Hugging Face model id or local checkpoint path.

    device:
        Optional device string, e.g. "cuda", "cpu", "cuda:0".
        If None, CUDA is used when available.

    trust_remote_code:
        Whether to trust custom model code from Hugging Face.

    model_kwargs:
        Extra arguments passed to AutoModelForSequenceClassification.

    Returns
    -------
    model, tokenizer, torch.device
    """
    resolved_device = get_device(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    model.to(resolved_device)
    model.eval()

    return model, tokenizer, resolved_device
