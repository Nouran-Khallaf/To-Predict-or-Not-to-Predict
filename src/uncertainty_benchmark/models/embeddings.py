"""Embedding extraction utilities."""

from __future__ import annotations

import numpy as np
import torch


def get_encoder_module(model):
    """Get the underlying transformer encoder from a classifier model.

    Supports common Hugging Face architectures:
    - BERT / mBERT
    - RoBERTa
    - DeBERTa
    - DistilBERT
    - models exposing `base_model`
    """
    for attr in ["bert", "roberta", "deberta", "distilbert", "xlm_roberta"]:
        if hasattr(model, attr):
            return getattr(model, attr)

    if hasattr(model, "base_model"):
        return model.base_model

    return model


@torch.no_grad()
def extract_cls_embeddings(
    model,
    tokenizer,
    texts,
    batch_size: int = 32,
    device=None,
    max_length: int | None = None,
) -> np.ndarray:
    """Extract CLS-style embeddings from the underlying encoder.

    For BERT-like models, this uses the first token representation:
        last_hidden_state[:, 0, :]

    Parameters
    ----------
    model:
        Hugging Face model.

    tokenizer:
        Matching tokenizer.

    texts:
        List of input strings.

    batch_size:
        Batch size for embedding extraction.

    device:
        Torch device. If None, inferred from model parameters.

    max_length:
        Optional tokenizer max length.

    Returns
    -------
    np.ndarray
        Array of shape [n_examples, hidden_size].
    """
    if device is None:
        device = next(model.parameters()).device

    encoder = get_encoder_module(model)
    model.eval()

    texts = list(texts)
    features = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        tok_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }

        if max_length is not None:
            tok_kwargs["max_length"] = max_length

        encoded = tokenizer(batch_texts, **tok_kwargs)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        outputs = encoder(**encoded, return_dict=True)

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs[0]

        cls = last_hidden_state[:, 0, :].detach().cpu().numpy()
        features.append(cls)

    if not features:
        return np.empty((0, 0))

    return np.concatenate(features, axis=0)
