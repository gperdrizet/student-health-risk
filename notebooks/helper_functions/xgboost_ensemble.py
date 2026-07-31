"""XGBoost helpers for sampled optimization and hill-climbing ensemble notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score

from helper_functions.gradient_boosting_baseline import sample_fold_split

try:
    from xgboost import XGBClassifier
    from xgboost.core import XGBoostError
except Exception:  # pragma: no cover - import guard for environments without xgboost
    XGBClassifier = None
    XGBoostError = Exception


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
        # XGBoost 2.x prefers device='cuda' with hist tree method.
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
):
    """Fit and predict with automatic GPU-to-CPU fallback if CUDA is unavailable."""

    model = build_xgb_model(params, seed=seed, prefer_gpu=prefer_gpu)

    try:
        model.fit(x_train, y_train)
    except XGBoostError:
        model = build_xgb_model(params, seed=seed, prefer_gpu=False)
        model.fit(x_train, y_train)

    predictions = model.predict(x_validation)
    return predictions


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
):
    """Evaluate one XGBoost parameter set across folds using balanced accuracy."""

    if sampling is None:
        sampling = FoldSamplingConfig()

    evaluation_folds = make_fixed_sampled_folds(folds, sampling)
    fold_scores = []

    for fold_index, fold in enumerate(evaluation_folds, start=1):
        y_pred = fit_predict_with_fallback(
            params,
            x_train=fold['x_train'],
            y_train=fold['y_train'],
            x_validation=fold['x_validation'],
            seed=seed + fold_index,
            prefer_gpu=prefer_gpu,
        )
        fold_scores.append(balanced_accuracy_score(fold['y_validation'], y_pred))

    return fold_scores


def bootstrap_rows_and_features(
    x_data,
    y_data,
    row_fraction,
    feature_fraction,
    seed,
):
    """Sample rows (stratified) and a random feature subset for model diversity."""

    if row_fraction is None:
        row_fraction = 1.0
    if feature_fraction is None:
        feature_fraction = 1.0

    x_rows, y_rows = sample_fold_split(
        x_data,
        y_data,
        sample_fraction=row_fraction,
        seed=seed,
    )

    if feature_fraction >= 1.0:
        selected_columns = list(x_rows.columns)
        return x_rows, y_rows, selected_columns

    if feature_fraction <= 0.0:
        raise ValueError('feature_fraction must be in the range (0, 1].')

    rng = np.random.default_rng(seed + 17)
    n_total = len(x_rows.columns)
    n_select = max(1, int(round(n_total * feature_fraction)))
    selected_columns = sorted(rng.choice(np.array(x_rows.columns), size=n_select, replace=False).tolist())

    return x_rows[selected_columns], y_rows, selected_columns


def average_probabilities(probability_matrices, weights=None):
    """Return weighted average probabilities for an ensemble."""

    if not probability_matrices:
        raise ValueError('probability_matrices cannot be empty.')

    matrix_count = len(probability_matrices)

    if weights is None:
        weights = np.repeat(1.0 / matrix_count, matrix_count)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    output = np.zeros_like(probability_matrices[0], dtype=float)

    for weight, probabilities in zip(weights, probability_matrices):
        output += weight * probabilities

    return output


def summarize_scores(fold_scores):
    """Return compact summary metrics from fold scores."""

    return {
        'mean': float(np.mean(fold_scores)),
        'median': float(np.median(fold_scores)),
        'std': float(np.std(fold_scores)),
        'fold_scores': list(map(float, fold_scores)),
    }
