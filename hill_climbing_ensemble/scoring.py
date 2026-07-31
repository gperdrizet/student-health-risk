"""Scoring and prediction routines for hill-climbing ensembles."""

from __future__ import annotations

import logging
import os
from multiprocessing import get_context
from queue import Empty
import traceback

import pandas as pd

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from .ml_utils import (
    XGBoostError,
    average_probabilities,
    bootstrap_rows_and_features,
    build_xgb_model,
    make_fixed_sampled_folds,
)


LOGGER = logging.getLogger('hill_climbing_ensemble.run')


def fit_model_from_spec(spec, x_train, y_train, fold_seed, prefer_gpu=True):
    row_seed = spec["proposal_seed"] + 100 * fold_seed

    x_bootstrap, y_bootstrap, _ = bootstrap_rows_and_features(
        x_train[spec["feature_columns"]],
        y_train,
        row_fraction=spec["row_fraction"],
        feature_fraction=1.0,
        seed=row_seed,
    )

    model = build_xgb_model(spec["params"], seed=row_seed, prefer_gpu=prefer_gpu)
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_bootstrap.to_numpy())

    try:
        model.fit(x_bootstrap, y_bootstrap, sample_weight=sample_weight)
    except XGBoostError:
        model = build_xgb_model(spec["params"], seed=row_seed, prefer_gpu=False)
        model.fit(x_bootstrap, y_bootstrap, sample_weight=sample_weight)

    return model


def _evaluate_one_fold(fold_index, fold, specs, seed):
    probability_matrices = []
    weights = []

    for spec in specs:
        model = fit_model_from_spec(
            spec,
            x_train=fold["x_train"],
            y_train=fold["y_train"],
            fold_seed=seed + fold_index,
            prefer_gpu=True,
        )

        probabilities = model.predict_proba(fold["x_validation"][spec["feature_columns"]])
        probability_matrices.append(probabilities)
        weights.append(spec.get("weight", 1.0))

    ensemble_probabilities = average_probabilities(probability_matrices, weights=weights)
    predictions = ensemble_probabilities.argmax(axis=1)
    return float(balanced_accuracy_score(fold["y_validation"], predictions))


def _partition_round_robin(items, worker_count):
    batches = [[] for _ in range(worker_count)]
    for item_index, item in enumerate(items):
        batches[item_index % worker_count].append(item)
    return batches


def _evaluate_fold_batch(worker_index, gpu_id, fold_batch, specs, seed, output_queue):
    try:
        # Pin this worker process to one visible GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        batch_rows = []

        for fold_index, fold in fold_batch:
            score = _evaluate_one_fold(
                fold_index=fold_index,
                fold=fold,
                specs=specs,
                seed=seed,
            )
            batch_rows.append(
                {
                    "fold_index": fold_index,
                    "score": score,
                    "worker_index": worker_index,
                    "gpu_id": gpu_id,
                }
            )

        output_queue.put({"ok": True, "rows": batch_rows, "worker_index": worker_index, "gpu_id": gpu_id})
    except Exception as exc:  # pragma: no cover - defensive against worker process failures
        output_queue.put(
            {
                "ok": False,
                "worker_index": worker_index,
                "gpu_id": gpu_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def evaluate_ensemble_specs(folds, specs, sampling_config, seed, gpu_ids=(0, 1)):
    if not specs:
        return []

    evaluation_folds = make_fixed_sampled_folds(folds, sampling_config)

    if len(gpu_ids) <= 1 or len(evaluation_folds) <= 1:
        return [
            _evaluate_one_fold(fold_index, fold, specs=specs, seed=seed)
            for fold_index, fold in enumerate(evaluation_folds, start=1)
        ]

    fold_pairs = list(enumerate(evaluation_folds, start=1))
    fold_batches = _partition_round_robin(fold_pairs, len(gpu_ids))

    # CUDA + fork can deadlock or silently fail in subprocesses; spawn is safer.
    ctx = get_context("spawn")
    output_queue = ctx.Queue()
    processes = []

    for worker_index, (gpu_id, batch) in enumerate(zip(gpu_ids, fold_batches), start=1):
        if not batch:
            continue

        process = ctx.Process(
            target=_evaluate_fold_batch,
            args=(worker_index, gpu_id, batch, specs, seed, output_queue),
        )
        process.start()
        processes.append(process)

    collected_rows = []
    try:
        for _ in processes:
            message = output_queue.get(timeout=1800)
            if not message.get("ok", False):
                raise RuntimeError(
                    "Parallel fold worker failed "
                    f"(worker={message.get('worker_index')}, gpu={message.get('gpu_id')}): "
                    f"{message.get('error')}\n{message.get('traceback', '')}"
                )
            collected_rows.extend(message["rows"])

        for process in processes:
            process.join()
            if process.exitcode not in (0, None):
                raise RuntimeError(f"Parallel fold worker exited with code {process.exitcode}")
    except (Empty, RuntimeError) as exc:
        # Avoid deadlocks if a worker crashes or never reports back.
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

        LOGGER.warning(
            "parallel_fold_eval_fallback reason=%s; switching to sequential evaluation",
            exc,
        )

        return [
            _evaluate_one_fold(fold_index, fold, specs=specs, seed=seed)
            for fold_index, fold in enumerate(evaluation_folds, start=1)
        ]

    rows = sorted(collected_rows, key=lambda row: row["fold_index"])
    return [row["score"] for row in rows]


def build_submission_from_specs(model_specs, train_df, test_df, label_encoder):
    if not model_specs:
        raise ValueError("Cannot build submission without model specs.")

    x_train = train_df.drop("health_condition", axis=1)
    y_train = train_df["health_condition"]

    probability_matrices = []
    weights = []

    for model_index, spec in enumerate(model_specs, start=1):
        model = fit_model_from_spec(
            spec,
            x_train=x_train,
            y_train=y_train,
            fold_seed=20000 + model_index,
            prefer_gpu=True,
        )

        probabilities = model.predict_proba(test_df[spec["feature_columns"]])
        probability_matrices.append(probabilities)
        weights.append(spec.get("weight", 1.0))

    ensemble_probabilities = average_probabilities(probability_matrices, weights=weights)
    predictions = ensemble_probabilities.argmax(axis=1)

    return pd.DataFrame(
        {
            "id": test_df["id"],
            "health_condition": label_encoder.inverse_transform(predictions.astype(int)),
        }
    )
