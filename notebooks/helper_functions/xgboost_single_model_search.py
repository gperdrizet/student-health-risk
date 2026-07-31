"""Helpers for single-model XGBoost search and rerank notebooks."""

from __future__ import annotations

import os
import time
from multiprocessing import get_context
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from helper_functions.gradient_boosting_baseline import sample_fold_split

try:
    from xgboost import XGBClassifier
    from xgboost.core import XGBoostError
except Exception:  # pragma: no cover - import guard for environments without xgboost
    XGBClassifier = None
    XGBoostError = Exception


_PARALLEL_CANDIDATE_FOLDS = None
_PARALLEL_CANDIDATE_SEED = None
_PARALLEL_USE_BALANCED_SAMPLE_WEIGHT = None
_PARALLEL_FOLD_PARAMS = None
_PARALLEL_FOLD_SEED = None
_PARALLEL_FOLD_USE_BALANCED_SAMPLE_WEIGHT = None


@dataclass
class FoldSamplingConfig:
    """Controls row sampling for fast CV at the fold-input level."""

    use_sampling: bool = False
    train_sample_fraction: float | None = None
    validation_sample_fraction: float | None = None
    sample_seed: int = 315


def _ensure_xgboost_available():
    if XGBClassifier is None:
        raise ImportError(
            'xgboost is not installed. Install it with `pip install xgboost` or via requirements.txt.'
        )


def build_xgb_model(params, seed=315, prefer_gpu=True):
    """Build an XGBClassifier with GPU-first config and CPU fallback support."""

    _ensure_xgboost_available()

    model_params = dict(params)
    model_params.setdefault('objective', 'multi:softmax')
    model_params.setdefault('eval_metric', 'mlogloss')
    model_params.setdefault('verbosity', 0)
    model_params.setdefault('random_state', seed)
    model_params.setdefault('n_jobs', -1)

    if prefer_gpu:
        model_params.setdefault('tree_method', 'hist')
        model_params.setdefault('device', 'cuda')
    else:
        model_params['tree_method'] = 'hist'
        model_params['device'] = 'cpu'

    return XGBClassifier(**model_params)


def fit_predict_with_fallback(
    params,
    x_train,
    y_train,
    x_validation,
    seed,
    prefer_gpu=True,
    sample_weight=None,
):
    """Fit and predict with automatic GPU-to-CPU fallback if CUDA is unavailable."""

    model = build_xgb_model(params, seed=seed, prefer_gpu=prefer_gpu)

    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except XGBoostError:
        model = build_xgb_model(params, seed=seed, prefer_gpu=False)
        model.fit(x_train, y_train, sample_weight=sample_weight)

    return model.predict(x_validation)


def make_fixed_sampled_folds(
    folds,
    sampling: FoldSamplingConfig,
):
    """Create a fixed sampled view of folds for fair candidate comparisons within one run."""

    if not sampling.use_sampling:
        return folds

    sampled_folds = []

    for fold_index, fold in enumerate(folds, start=1):
        sampled_fold = dict(fold)

        x_train_sampled, y_train_sampled = sample_fold_split(
            fold['x_train'],
            fold['y_train'],
            sample_fraction=sampling.train_sample_fraction,
            seed=sampling.sample_seed + 1000 + fold_index,
        )

        x_validation_sampled, y_validation_sampled = sample_fold_split(
            fold['x_validation'],
            fold['y_validation'],
            sample_fraction=sampling.validation_sample_fraction,
            seed=sampling.sample_seed + 2000 + fold_index,
        )

        sampled_fold['x_train'] = x_train_sampled
        sampled_fold['y_train'] = y_train_sampled
        sampled_fold['x_validation'] = x_validation_sampled
        sampled_fold['y_validation'] = y_validation_sampled

        sampled_folds.append(sampled_fold)

    return sampled_folds


def score_xgb_on_folds(
    folds,
    params,
    sampling: FoldSamplingConfig | None = None,
    seed=315,
    prefer_gpu=True,
    use_balanced_sample_weight=False,
):
    """Evaluate one XGBoost parameter set across folds using balanced accuracy."""

    if sampling is None:
        sampling = FoldSamplingConfig()

    evaluation_folds = make_fixed_sampled_folds(folds, sampling)
    fold_scores = []

    for fold_index, fold in enumerate(evaluation_folds, start=1):
        sample_weight = None
        if use_balanced_sample_weight:
            sample_weight = compute_sample_weight(
                class_weight='balanced',
                y=np.asarray(fold['y_train']),
            )

        y_pred = fit_predict_with_fallback(
            params,
            x_train=fold['x_train'],
            y_train=fold['y_train'],
            x_validation=fold['x_validation'],
            seed=seed + fold_index,
            prefer_gpu=prefer_gpu,
            sample_weight=sample_weight,
        )
        fold_scores.append(balanced_accuracy_score(fold['y_validation'], y_pred))

    return fold_scores


def summarize_scores(fold_scores):
    """Return compact summary metrics from fold scores."""

    return {
        'mean': float(np.mean(fold_scores)),
        'median': float(np.median(fold_scores)),
        'std': float(np.std(fold_scores)),
        'fold_scores': list(map(float, fold_scores)),
    }


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
            use_balanced_sample_weight=_PARALLEL_USE_BALANCED_SAMPLE_WEIGHT,
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


def score_xgb_candidates_parallel(candidates, folds, seed, gpu_ids, use_balanced_sample_weight=False):
    global _PARALLEL_CANDIDATE_FOLDS, _PARALLEL_CANDIDATE_SEED, _PARALLEL_USE_BALANCED_SAMPLE_WEIGHT

    if len(gpu_ids) <= 1 or len(candidates) <= 1:
        rows = []

        for candidate_index, params in enumerate(candidates, start=1):
            fold_scores = score_xgb_on_folds(
                folds,
                params,
                sampling=FoldSamplingConfig(use_sampling=False),
                seed=seed,
                prefer_gpu=True,
                use_balanced_sample_weight=use_balanced_sample_weight,
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
    _PARALLEL_USE_BALANCED_SAMPLE_WEIGHT = use_balanced_sample_weight

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
        sample_weight = None
        if _PARALLEL_FOLD_USE_BALANCED_SAMPLE_WEIGHT:
            sample_weight = compute_sample_weight(
                class_weight='balanced',
                y=np.asarray(fold['y_train']),
            )

        y_pred = fit_predict_with_fallback(
            _PARALLEL_FOLD_PARAMS,
            x_train=fold['x_train'],
            y_train=fold['y_train'],
            x_validation=fold['x_validation'],
            seed=_PARALLEL_FOLD_SEED + fold_index,
            prefer_gpu=True,
            sample_weight=sample_weight,
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


def score_xgb_on_folds_parallel(folds, params, seed, gpu_ids, use_balanced_sample_weight=False):
    global _PARALLEL_FOLD_PARAMS, _PARALLEL_FOLD_SEED, _PARALLEL_FOLD_USE_BALANCED_SAMPLE_WEIGHT

    if len(gpu_ids) <= 1 or len(folds) <= 1:
        return score_xgb_on_folds(
            folds,
            params,
            sampling=FoldSamplingConfig(use_sampling=False),
            seed=seed,
            prefer_gpu=True,
            use_balanced_sample_weight=use_balanced_sample_weight,
        )

    _PARALLEL_FOLD_PARAMS = params
    _PARALLEL_FOLD_SEED = seed
    _PARALLEL_FOLD_USE_BALANCED_SAMPLE_WEIGHT = use_balanced_sample_weight

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
