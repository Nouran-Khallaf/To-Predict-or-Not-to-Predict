"""Model utilities."""

from uncertainty_benchmark.models.dropout import (
    DropoutMC,
    activate_mc_dropout,
    convert_dropouts,
    convert_to_mc_dropout,
    count_mc_dropout_layers,
)
from uncertainty_benchmark.models.embeddings import (
    extract_cls_embeddings,
    get_encoder_module,
)
from uncertainty_benchmark.models.hf_loader import (
    get_device,
    load_model_and_tokenizer,
)
from uncertainty_benchmark.models.predictors import (
    build_trainer,
    majority_vote_predictions,
    predict_logits_probs_labels,
)

__all__ = [
    "DropoutMC",
    "activate_mc_dropout",
    "convert_dropouts",
    "convert_to_mc_dropout",
    "count_mc_dropout_layers",
    "extract_cls_embeddings",
    "get_encoder_module",
    "get_device",
    "load_model_and_tokenizer",
    "build_trainer",
    "majority_vote_predictions",
    "predict_logits_probs_labels",
]
