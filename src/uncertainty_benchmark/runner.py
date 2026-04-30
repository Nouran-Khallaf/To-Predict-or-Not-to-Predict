"""Main experiment runner.

This module connects the clean layers:

- data loading
- Hugging Face model prediction
- uncertainty scoring
- metrics
- timing
- saving and aggregation
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from uncertainty_benchmark.data import (
    build_label_encoder,
    dataframe_to_texts_labels,
    load_eval_from_prediction_csv,
    load_train_file,
    remove_text_overlap,
    tokenize_dataframe,
)
from uncertainty_benchmark.io import (
    ensure_dir,
    save_dataframe,
    scores_wide_to_long,
    summarise_method_metric_times,
    summarise_numeric_columns,
    summarise_total_times,
)
from uncertainty_benchmark.metrics import (
    compute_metrics_per_method_with_timing,
    metrics_to_long,
)
from uncertainty_benchmark.methods import build_method
from uncertainty_benchmark.models import (
    activate_mc_dropout,
    build_trainer,
    convert_dropouts,
    extract_cls_embeddings,
    load_model_and_tokenizer,
    majority_vote_predictions,
    predict_logits_probs_labels,
)
from uncertainty_benchmark.seed import seed_everything
from uncertainty_benchmark.timing import Timer


METHOD_ORDER = [
    "SR",
    "SMP",
    "ENT",
    "ENT_MC",
    "PV",
    "BALD",
    "MD",
    "HUQ-MD",
    "LOF",
    "ISOF",
]

MC_METHODS = {"SMP", "PV", "BALD", "ENT_MC"}
OUTLIER_METHODS = {"LOF", "ISOF"}


def method_key(method_name: str) -> str:
    """Safe key for timing column names."""
    return method_name.lower().replace("-", "_")


def resolve_fold_ids(config: dict) -> list[int]:
    """Resolve fold ids from config.

    Priority:
    1. folds.fold_ids if present
    2. range(folds.n_folds)
    """
    folds = config.get("folds", {})

    if "fold_ids" in folds and folds["fold_ids"] is not None:
        return [int(x) for x in folds["fold_ids"]]

    n_folds = int(folds.get("n_folds", 1))
    return list(range(n_folds))


def get_enabled_methods(config: dict) -> list[str]:
    """Return enabled methods in stable order."""
    enabled = config.get("methods", {}).get("enabled", METHOD_ORDER)
    enabled_set = set(enabled)
    return [m for m in METHOD_ORDER if m in enabled_set]


def get_required_methods(enabled_methods: list[str]) -> list[str]:
    """Add hidden dependencies needed to compute enabled methods."""
    required = set(enabled_methods)

    if "HUQ-MD" in required:
        required.add("SR")
        required.add("MD")

    return [m for m in METHOD_ORDER if m in required]


def make_output_dirs(outdir) -> dict:
    """Create structured output directories."""
    outdir = ensure_dir(outdir)

    dirs = {
        "root": outdir,
        "scores": ensure_dir(outdir / "scores"),
        "metrics": ensure_dir(outdir / "metrics"),
        "timing": ensure_dir(outdir / "timing"),
        "logs": ensure_dir(outdir / "logs"),
    }

    return dirs


def compute_method_score(method_name: str, context: dict, timings: dict):
    """Compute one uncertainty method score and time it."""
    timer = Timer()
    timer.start()

    method = build_method(method_name)
    score = method.score(context)

    elapsed = timer.stop()
    timings[f"compute_{method_key(method_name)}_s"] = elapsed

    context[method_name] = score
    return score


def compute_mc_dropout_predictions(
    trainer,
    eval_dataset,
    committee_size: int,
    dropout_p: float,
    model,
    timings: dict,
):
    """Compute MC-dropout sampled probabilities and sampled predictions."""
    timer = Timer()
    timer.start()

    convert_dropouts(model, inference_prob=dropout_p)
    activate_mc_dropout(model, activate=True)

    sampled_prob_list = []
    sampled_pred_list = []

    for _ in range(committee_size):
        out_i = predict_logits_probs_labels(trainer, eval_dataset)
        probs_i = out_i["probs"]
        preds_i = out_i["y_pred"]

        sampled_prob_list.append(probs_i)
        sampled_pred_list.append(preds_i)

    activate_mc_dropout(model, activate=False)

    timings["mc_dropout_predict_block_s"] = timer.stop()

    sampled_probs = np.stack(sampled_prob_list, axis=1)
    sampled_preds = np.stack(sampled_pred_list, axis=1)

    timer = Timer()
    timer.start()
    vote_preds = majority_vote_predictions(sampled_preds)
    timings["compute_final_preds_vote_s"] = timer.stop()

    return sampled_probs, sampled_preds, vote_preds


def build_scores_dataframe(
    fold_id: int,
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder,
    enabled_methods: list[str],
    scores: dict,
) -> pd.DataFrame:
    """Create per-sample wide score dataframe."""
    df_out = pd.DataFrame(
        {
            "fold": fold_id,
            "text": texts,
            "y_true_idx": y_true.astype(int),
            "y_pred_idx": y_pred.astype(int),
            "true_label": encoder.inverse_transform(y_true.astype(int)),
            "predicted_label": encoder.inverse_transform(y_pred.astype(int)),
            "correct": y_true.astype(int) == y_pred.astype(int),
        }
    )

    for method in enabled_methods:
        df_out[method] = scores[method]

    return df_out


def compute_standalone_method_times(
    fold_id: int,
    n_eval: int,
    enabled_methods: list[str],
    timings: dict,
    method_metric_times: dict,
) -> pd.DataFrame:
    """Compute standalone total time per method.

    This follows the original experiment logic:

    total_s(method) = uncertainty_standalone_s(method) + metric_compute_s(method)

    Important:
    - SR/ENT include base prediction time.
    - MC methods include the MC-dropout prediction block.
    - MD/HUQ-MD include train-logit prediction.
    - LOF/ISOF include embedding extraction and outlier fit/score.
    """
    rows = []

    base = float(timings.get("predict_eval_base_s", 0.0))
    train_logits = float(timings.get("predict_train_logits_s", 0.0))
    mc_block = float(timings.get("mc_dropout_predict_block_s", 0.0))
    train_emb = float(timings.get("extract_train_emb_s", 0.0))
    eval_emb = float(timings.get("extract_eval_emb_s", 0.0))

    for method in enabled_methods:
        key = method_key(method)
        compute_time = float(timings.get(f"compute_{key}_s", 0.0))

        if method == "SR":
            uncertainty_s = base + compute_time

        elif method == "ENT":
            uncertainty_s = base + compute_time

        elif method in MC_METHODS:
            uncertainty_s = mc_block + compute_time

        elif method == "MD":
            uncertainty_s = base + train_logits + compute_time

        elif method == "HUQ-MD":
            sr_time = float(timings.get("compute_sr_s", 0.0))
            md_time = float(timings.get("compute_md_s", 0.0))
            uncertainty_s = base + sr_time + train_logits + md_time + compute_time

        elif method in OUTLIER_METHODS:
            uncertainty_s = train_emb + eval_emb + compute_time

        else:
            uncertainty_s = compute_time

        metrics_s = float(method_metric_times[method])
        total_s = uncertainty_s + metrics_s

        rows.append(
            {
                "fold": fold_id,
                "n_eval": int(n_eval),
                "method": method,
                "uncertainty_s": uncertainty_s,
                "metrics_s": metrics_s,
                "total_s": total_s,
                "total_ms_per_ex": (total_s / float(n_eval)) * 1000.0,
                "ex_per_s": float(n_eval) / total_s if total_s > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def run_fold(config: dict, fold_id: int):
    """Run one fold and return all fold-level outputs."""
    seed_everything(42 + int(fold_id))
    os.environ["WANDB_DISABLED"] = "true"

    enabled_methods = get_enabled_methods(config)
    required_methods = get_required_methods(enabled_methods)

    outdir = Path(config["outputs"]["outdir"])
    dirs = make_output_dirs(outdir)

    timings = {"fold": int(fold_id)}

    total_timer = Timer()
    total_timer.start()

    # -------------------------
    # Load model and tokenizer
    # -------------------------
    timer = Timer()
    timer.start()

    model_id = config["model"]["model_id_template"].format(fold_id=fold_id)
    model, tokenizer, device = load_model_and_tokenizer(model_id)

    timings["load_model_tokenizer_s"] = timer.stop()

    # -------------------------
    # Load data
    # -------------------------
    encoder = build_label_encoder(config["labels"]["classes"])

    timer = Timer()
    timer.start()
    df_eval = load_eval_from_prediction_csv(config, fold_id=fold_id, encoder=encoder)
    timings["load_eval_from_predcsv_s"] = timer.stop()

    timer = Timer()
    timer.start()
    df_train = load_train_file(config, encoder=encoder)
    df_train = remove_text_overlap(df_train, df_eval, train_text_col="text", eval_text_col="text")
    timings["load_train_prepare_s"] = timer.stop()

    eval_texts, _ = dataframe_to_texts_labels(df_eval)
    train_texts, train_labels = dataframe_to_texts_labels(df_train)

    timings["n_eval"] = int(len(df_eval))
    timings["n_train"] = int(len(df_train))

    # -------------------------
    # Tokenize
    # -------------------------
    timer = Timer()
    timer.start()
    eval_dataset = tokenize_dataframe(df_eval, tokenizer)
    train_dataset = tokenize_dataframe(df_train, tokenizer)
    timings["tokenize_data_s"] = timer.stop()

    # -------------------------
    # Trainer
    # -------------------------
    timer = Timer()
    timer.start()
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    timings["init_trainer_s"] = timer.stop()

    # -------------------------
    # Base prediction
    # -------------------------
    timer = Timer()
    timer.start()
    base = predict_logits_probs_labels(trainer, eval_dataset)
    timings["predict_eval_base_s"] = timer.stop()

    eval_logits = base["logits"]
    eval_probs = base["probs"]
    y_true = base["y_true"]

    if y_true is None:
        y_true = df_eval["labels"].astype(int).values

    y_pred = base["y_pred"]

    context = {
        "eval_logits": eval_logits,
        "eval_probs": eval_probs,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    # -------------------------
    # MC-dropout shared block
    # -------------------------
    if any(method in required_methods for method in MC_METHODS):
        committee_size = int(config.get("mc_dropout", {}).get("committee_size", 20))
        dropout_p = float(config.get("mc_dropout", {}).get("dropout_p", 0.10))

        sampled_probs, sampled_preds, vote_preds = compute_mc_dropout_predictions(
            trainer=trainer,
            eval_dataset=eval_dataset,
            committee_size=committee_size,
            dropout_p=dropout_p,
            model=model,
            timings=timings,
        )

        context["sampled_probs"] = sampled_probs
        context["sampled_preds"] = sampled_preds
        context["vote_preds"] = vote_preds

    # -------------------------
    # Train logits for MD/HUQ-MD
    # -------------------------
    if "MD" in required_methods or "HUQ-MD" in required_methods:
        timer = Timer()
        timer.start()
        train_base = predict_logits_probs_labels(trainer, train_dataset)
        timings["predict_train_logits_s"] = timer.stop()

        context["train_logits"] = train_base["logits"]
        context["train_labels"] = train_labels
        context["eval_logits"] = eval_logits

    # -------------------------
    # Embeddings for LOF/ISOF
    # -------------------------
    if any(method in required_methods for method in OUTLIER_METHODS):
        batch_size = int(config.get("embeddings", {}).get("batch_size", 32))

        timer = Timer()
        timer.start()
        train_emb = extract_cls_embeddings(
            model,
            tokenizer,
            train_texts,
            batch_size=batch_size,
            device=device,
        )
        timings["extract_train_emb_s"] = timer.stop()

        timer = Timer()
        timer.start()
        eval_emb = extract_cls_embeddings(
            model,
            tokenizer,
            eval_texts,
            batch_size=batch_size,
            device=device,
        )
        timings["extract_eval_emb_s"] = timer.stop()

        context["train_emb"] = train_emb
        context["eval_emb"] = eval_emb

    # -------------------------
    # Compute uncertainty scores
    # -------------------------
    scores = {}

    for method in required_methods:
        score = compute_method_score(method, context, timings)

        if method in enabled_methods:
            scores[method] = score

    # -------------------------
    # Save per-sample scores
    # -------------------------
    timer = Timer()
    timer.start()

    # Original-notebook compatibility:
    # The notebook saved final_preds from MC-dropout majority vote when MC
    # dropout was run, then computed correctness and selective metrics from
    # those final predictions. If no MC vote is available, fall back to the
    # deterministic/base prediction. This can be overridden with:
    #   metrics:
    #     prediction_source: base
    # or
    #   metrics:
    #     prediction_source: mc_majority
    prediction_source = config.get("metrics", {}).get(
        "prediction_source",
        "mc_majority_if_available",
    )

    if prediction_source == "base":
        y_pred_for_metrics = y_pred
    elif prediction_source == "mc_majority":
        if "vote_preds" not in context:
            raise ValueError(
                "metrics.prediction_source is 'mc_majority', but MC-dropout "
                "was not run, so vote_preds are unavailable."
            )
        y_pred_for_metrics = context["vote_preds"]
    elif prediction_source == "mc_majority_if_available":
        y_pred_for_metrics = context.get("vote_preds", y_pred)
    else:
        raise ValueError(
            "Unknown metrics.prediction_source: "
            f"{prediction_source}. Use 'base', 'mc_majority', or "
            "'mc_majority_if_available'."
        )

    df_scores = build_scores_dataframe(
        fold_id=fold_id,
        texts=eval_texts,
        y_true=y_true,
        y_pred=y_pred_for_metrics,
        encoder=encoder,
        enabled_methods=enabled_methods,
        scores=scores,
    )

    df_scores["base_y_pred_idx"] = y_pred.astype(int)
    df_scores["base_predicted_label"] = encoder.inverse_transform(y_pred.astype(int))
    df_scores["prediction_source_for_metrics"] = prediction_source

    if config.get("outputs", {}).get("save_wide_scores", True):
        save_dataframe(
            df_scores,
            dirs["scores"] / f"fold_{fold_id}_scores_wide.csv",
            index=False,
        )

    if config.get("outputs", {}).get("save_long_scores", True):
        df_scores_long = scores_wide_to_long(df_scores, enabled_methods)
        save_dataframe(
            df_scores_long,
            dirs["scores"] / f"fold_{fold_id}_scores_long.csv",
            index=False,
        )

    timings["save_uncertainty_scores_s"] = timer.stop()

    # -------------------------
    # Metrics
    # -------------------------
    timer = Timer()
    timer.start()

    bins = int(config.get("metrics", {}).get("ece_bins", 15))
    ti_fixed_coverage = float(config.get("metrics", {}).get("ti_fixed_coverage", 0.95))

    metrics_df, method_metric_times = compute_metrics_per_method_with_timing(
        df_scores,
        methods=enabled_methods,
        bins=bins,
        ti_fixed_coverage=ti_fixed_coverage,
    )

    timings["compute_all_metrics_total_s"] = timer.stop()

    metrics_path = dirs["metrics"] / f"fold_{fold_id}_metrics.csv"
    save_dataframe(metrics_df, metrics_path, index=True)

    metrics_long = metrics_to_long(metrics_df, fold=fold_id)
    save_dataframe(
        metrics_long,
        dirs["metrics"] / f"fold_{fold_id}_metrics_long.csv",
        index=False,
    )

    method_metric_times_df = pd.DataFrame([method_metric_times])
    method_metric_times_df["fold"] = int(fold_id)
    save_dataframe(
        method_metric_times_df,
        dirs["timing"] / f"fold_{fold_id}_method_metric_times.csv",
        index=False,
    )

    # -------------------------
    # Standalone method total times
    # -------------------------
    method_total_times_df = compute_standalone_method_times(
        fold_id=fold_id,
        n_eval=len(df_eval),
        enabled_methods=enabled_methods,
        timings=timings,
        method_metric_times=method_metric_times,
    )

    save_dataframe(
        method_total_times_df,
        dirs["timing"] / f"fold_{fold_id}_method_total_times.csv",
        index=False,
    )

    timings["total_fold_s"] = total_timer.stop()

    timings_df = pd.DataFrame([timings])
    save_dataframe(
        timings_df,
        dirs["timing"] / f"fold_{fold_id}_block_timings.csv",
        index=False,
    )

    print(
        f"[Fold {fold_id}] complete | "
        f"N_eval={timings['n_eval']} | "
        f"N_train={timings['n_train']} | "
        f"total={timings['total_fold_s'] / 60.0:.2f} min"
    )

    return {
        "timings": timings,
        "metrics_long": metrics_long,
        "method_metric_times": method_metric_times_df,
        "method_total_times": method_total_times_df,
    }


def run_experiment(config: dict):
    """Run all folds and save aggregate outputs."""
    fold_ids = resolve_fold_ids(config)
    enabled_methods = get_enabled_methods(config)

    outdir = Path(config["outputs"]["outdir"])
    dirs = make_output_dirs(outdir)

    print("Running experiment")
    print(f"  output: {outdir}")
    print(f"  folds: {fold_ids}")
    print(f"  methods: {enabled_methods}")

    all_timings = []
    all_metrics_long = []
    all_method_metric_times = []
    all_method_total_times = []

    for fold_id in fold_ids:
        print(f"\n===== Running fold {fold_id} =====")

        result = run_fold(config, fold_id)

        all_timings.append(result["timings"])
        all_metrics_long.append(result["metrics_long"])
        all_method_metric_times.append(result["method_metric_times"])
        all_method_total_times.append(result["method_total_times"])

    # -------------------------
    # Aggregate block timings
    # -------------------------
    timings_all = pd.DataFrame(all_timings)
    save_dataframe(
        timings_all,
        dirs["timing"] / "timings_blocks_all_folds.csv",
        index=False,
    )

    timings_summary = summarise_numeric_columns(timings_all, id_columns=["fold"])
    save_dataframe(
        timings_summary,
        dirs["timing"] / "timings_blocks_summary_mean_std.csv",
        index=False,
    )

    # -------------------------
    # Aggregate metrics
    # -------------------------
    metrics_long_all = pd.concat(all_metrics_long, ignore_index=True)
    save_dataframe(
        metrics_long_all,
        dirs["metrics"] / "metrics_long_all_folds.csv",
        index=False,
    )

    metrics_summary = (
        metrics_long_all
        .groupby(["method", "metric"])
        .agg(
            folds=("fold", "nunique"),
            mean=("value", "mean"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
        )
        .reset_index()
    )
    save_dataframe(
        metrics_summary,
        dirs["metrics"] / "metrics_summary_mean_std.csv",
        index=False,
    )

    # -------------------------
    # Aggregate metric computation times
    # -------------------------
    method_metric_times_all = pd.concat(all_method_metric_times, ignore_index=True)
    save_dataframe(
        method_metric_times_all,
        dirs["timing"] / "method_metric_times_all_folds.csv",
        index=False,
    )

    method_metric_times_summary = summarise_method_metric_times(method_metric_times_all)
    save_dataframe(
        method_metric_times_summary,
        dirs["timing"] / "method_metric_times_summary_mean_std.csv",
        index=False,
    )

    # -------------------------
    # Aggregate standalone total times
    # -------------------------
    method_total_times_all = pd.concat(all_method_total_times, ignore_index=True)
    save_dataframe(
        method_total_times_all,
        dirs["timing"] / "method_total_times_all_folds.csv",
        index=False,
    )

    method_total_times_summary = summarise_total_times(method_total_times_all)
    save_dataframe(
        method_total_times_summary,
        dirs["timing"] / "method_total_times_summary_mean_std.csv",
        index=False,
    )

    print("\nSaved aggregate outputs to:")
    print(f"  {dirs['root']}")
