"""Prediction helpers for Hugging Face Trainer-based inference."""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from transformers import DataCollatorWithPadding, Trainer


def build_trainer(
    model,
    tokenizer,
    train_dataset=None,
    eval_dataset=None,
    **trainer_kwargs,
) -> Trainer:
    """Build a lightweight Hugging Face Trainer for inference."""
    return Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        **trainer_kwargs,
    )


def predict_logits_probs_labels(trainer: Trainer, dataset):
    """Predict logits, probabilities, gold labels, and predicted labels.

    Returns
    -------
    dict
        - logits
        - probs
        - y_true
        - y_pred
    """
    output = trainer.predict(dataset)

    logits = output.predictions
    probs = softmax(logits, axis=1)

    y_true = None
    if output.label_ids is not None:
        y_true = np.asarray(output.label_ids, dtype=int)

    y_pred = probs.argmax(axis=-1).astype(int)

    return {
        "logits": logits,
        "probs": probs,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def majority_vote_predictions(sampled_preds: np.ndarray) -> np.ndarray:
    """Majority vote over sampled predictions.

    sampled_preds shape:
        [n_examples, n_samples]
    """
    sampled_preds = np.asarray(sampled_preds, dtype=int)

    if sampled_preds.ndim != 2:
        raise ValueError(
            "sampled_preds must have shape [n_examples, n_samples]. "
            f"Got shape {sampled_preds.shape}."
        )

    return np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=1,
        arr=sampled_preds,
    )
