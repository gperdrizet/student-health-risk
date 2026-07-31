"""Helpers for single-model XGBoost search and rerank notebooks."""

from __future__ import annotations

import os
import time
from multiprocessing import get_context

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from helper_functions.xgboost_ensemble import (
    FoldSamplingConfig,
    fit_predict_with_fallback,
    score_xgb_on_folds,
    summarize_scores,
)


_PARALLEL_CANDIDATE_FOLDS = None
_PARALLEL_CANDIDATE_SEED = None
_PARALLEL_FOLD_PARAMS = None
_PARALLEL_FOLD_SEED = None


def limited_folds(folds, fold_limit):
    """Return all folds or the first fold_limit folds."""

    if fold_limit is None or fold_limit <= 0:
        return folds
    return folds[:min(fold_limit, len(folds))]


def build_relative_xgb_candidate(base_params, spec):
    """Build one candidate by applying relative deltas/scales to base params."""

    candidate = dict(base_params)

    if 'learning_rate_mult' in spec:
        candidate['learning_rate'] = max(1e-4, base_params['learning_rate'] * spec['learning_rate_mult'])

    if 'n_estimators_mult' in spec:
        candidate['n_estimators'] = max(80, int(round(base_params['n_estimators'] * spec['n_estimators_mult'])))

    if 'max_depth_delta' in spec:
        candidate['max_depth'] = max(2, int(round(base_params['max_depth'] + spec['max_depth_delta'])))

    if 'subsample_mult' in spec:
        candidate['subsample'] = min(1.0, max(0.3, base_params['subsample'] * spec['subsample_mult']))

    if 'colsample_mult' in spec:
        candidate['colsample_bytree'] = min(1.0, max(0.3, base_params['colsample_bytree'] * spec['colsample_mult']))

    return candidate


def derive_parameter_ranges(rows, top_n=8):
    """Derive parameter min/max ranges from top rows."""

    selected_rows = rows[:top_n]
    ranges = {}

    for key in selected_rows[0]['params'].keys():
        values = [row['params'][key] for row in selected_rows]

        if isinstance(values[0], int):
            ranges[key] = [int(min(values)), int(max(values))]
        else:
            ranges[key] = [float(min(values)), float(max(values))]

    return ranges


def partition_round_robin(items, worker_count):
    """Distribute items across workers in round-robin order."""

    batches = [[] for _ in range(worker_count)]

    for item_index, item in enumerate(items):
        batches[item_index % worker_count].append(item)

    return batches


def format_duration(seconds):
    """Format elapsed seconds into a compact human-readable string."""

    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return 'unknown'

    seconds = int(round(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)

    if hours:
        return f'{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s'

    if minutes:
        return f'{minutes}m {remaining_seconds:02d}s'

    return f'{remaining_seconds}s'


def _evaluate_candidate_batch(worker_index, gpu_id, candidate_batch, output_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    batch_rows = []
    batch_start = time.time()

    print(f'  worker {worker_index} on GPU {gpu_id}: {len(candidate_batch)} candidates')

    for candidate_index, params in candidate_batch:
        fold_scores = score_xgb_on_folds(
            _PARALLEL_CANDIDATE_FOLDS,
            params,
            sampling=FoldSamplingConfig(use_sampling=False),
            seed=_PARALLEL_CANDIDATE_SEED,
            prefer_gpu=True,
        )

        row = summarize_scores(fold_scores)
        row['params'] = params
        row['candidate_index'] = candidate_index
        row['worker_index'] = worker_index
        row['gpu_id'] = gpu_id
        batch_rows.append(row)

        completed = len(batch_rows)
        if completed % 10 == 0 or completed == len(candidate_batch):
            elapsed_seconds = time.time() - batch_start
            rate = completed / elapsed_seconds if elapsed_seconds > 0 else np.nan
            remaining = len(candidate_batch) - completed
            eta_seconds = remaining / rate if rate and np.isfinite(rate) and rate > 0 else np.nan
            print(
                f'  worker {worker_index} on GPU {gpu_id}: '
                f'{completed}/{len(candidate_batch)} candidates, '
                f'elapsed={format_duration(elapsed_seconds)}, '
                f'eta={format_duration(eta_seconds)}'
            )

    output_queue.put(batch_rows)


def score_xgb_candidates_parallel(candidates, folds, seed, gpu_ids):
    global _PARALLEL_CANDIDATE_FOLDS, _PARALLEL_CANDIDATE_SEED

    if len(gpu_ids) <= 1 or len(candidates) <= 1:
        rows = []

        for candidate_index, params in enumerate(candidates, start=1):
            fold_scores = score_xgb_on_folds(
                folds,
                params,
                sampling=FoldSamplingConfig(use_sampling=False),
                seed=seed,
                prefer_gpu=True,
            )

            row = summarize_scores(fold_scores)
            row['params'] = params
            row['candidate_index'] = candidate_index
            row['worker_index'] = 1
            row['gpu_id'] = gpu_ids[0] if gpu_ids else 0
            rows.append(row)

        return rows

    _PARALLEL_CANDIDATE_FOLDS = folds
    _PARALLEL_CANDIDATE_SEED = seed

    candidate_pairs = list(enumerate(candidates, start=1))
    candidate_batches = partition_round_robin(candidate_pairs, len(gpu_ids))
    print(
        f'Starting parallel candidate scoring: {len(candidates)} candidates, '
        f'{len(folds)} folds each, ~{len(candidates) * len(folds)} fits total across {len(gpu_ids)} GPUs'
    )
    ctx = get_context('fork')
    output_queue = ctx.Queue()
    processes = []
    wall_start = time.time()

    for worker_index, (gpu_id, batch) in enumerate(zip(gpu_ids, candidate_batches), start=1):
        if not batch:
            continue

        process = ctx.Process(
            target=_evaluate_candidate_batch,
            args=(worker_index, gpu_id, batch, output_queue),
        )
        process.start()
        processes.append(process)

    collected_rows = []

    for _ in processes:
        collected_rows.extend(output_queue.get())

    for process in processes:
        process.join()

    rows = sorted(collected_rows, key=lambda row: row['candidate_index'])
    elapsed_minutes = (time.time() - wall_start) / 60.0
    print(f'Parallel candidate scoring used {len(processes)} GPU workers in {elapsed_minutes:.2f} min')

    return rows


def _evaluate_fold_batch(worker_index, gpu_id, fold_batch, output_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    batch_rows = []
    batch_start = time.time()

    print(f'  worker {worker_index} on GPU {gpu_id}: {len(fold_batch)} folds')

    for fold_index, fold in fold_batch:
        y_pred = fit_predict_with_fallback(
            _PARALLEL_FOLD_PARAMS,
            x_train=fold['x_train'],
            y_train=fold['y_train'],
            x_validation=fold['x_validation'],
            seed=_PARALLEL_FOLD_SEED + fold_index,
            prefer_gpu=True,
        )
        batch_rows.append({
            'fold_index': fold_index,
            'score': float(balanced_accuracy_score(fold['y_validation'], y_pred)),
            'worker_index': worker_index,
            'gpu_id': gpu_id,
        })

        completed = len(batch_rows)
        if completed % 2 == 0 or completed == len(fold_batch):
            elapsed_seconds = time.time() - batch_start
            rate = completed / elapsed_seconds if elapsed_seconds > 0 else np.nan
            remaining = len(fold_batch) - completed
            eta_seconds = remaining / rate if rate and np.isfinite(rate) and rate > 0 else np.nan
            print(
                f'  worker {worker_index} on GPU {gpu_id}: '
                f'{completed}/{len(fold_batch)} folds, '
                f'elapsed={format_duration(elapsed_seconds)}, '
                f'eta={format_duration(eta_seconds)}'
            )

    output_queue.put(batch_rows)


def score_xgb_on_folds_parallel(folds, params, seed, gpu_ids):
    global _PARALLEL_FOLD_PARAMS, _PARALLEL_FOLD_SEED

    if len(gpu_ids) <= 1 or len(folds) <= 1:
        return score_xgb_on_folds(
            folds,
            params,
            sampling=FoldSamplingConfig(use_sampling=False),
            seed=seed,
            prefer_gpu=True,
        )

    _PARALLEL_FOLD_PARAMS = params
    _PARALLEL_FOLD_SEED = seed

    fold_pairs = list(enumerate(folds, start=1))
    fold_batches = partition_round_robin(fold_pairs, len(gpu_ids))
    print(
        f'Starting parallel fold scoring: {len(folds)} folds, '
        f'~{len(folds)} fits total across {len(gpu_ids)} GPUs'
    )
    ctx = get_context('fork')
    output_queue = ctx.Queue()
    processes = []
    wall_start = time.time()

    for worker_index, (gpu_id, batch) in enumerate(zip(gpu_ids, fold_batches), start=1):
        if not batch:
            continue

        process = ctx.Process(
            target=_evaluate_fold_batch,
            args=(worker_index, gpu_id, batch, output_queue),
        )
        process.start()
        processes.append(process)

    collected_rows = []

    for _ in processes:
        collected_rows.extend(output_queue.get())

    for process in processes:
        process.join()

    rows = sorted(collected_rows, key=lambda row: row['fold_index'])
    elapsed_minutes = (time.time() - wall_start) / 60.0
    print(f'Parallel fold scoring used {len(processes)} GPU workers in {elapsed_minutes:.2f} min')

    return [row['score'] for row in rows]
