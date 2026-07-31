"""Runtime-local ML helpers for hill-climbing ensemble workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score

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


def _ensure_xgboost_available() -> None:
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


def sample_fold_split(x_data, y_data, sample_fraction, seed):
    """Return a stratified subsample of one fold split using a row fraction."""

    if sample_fraction is None or sample_fraction >= 1.0:
        return x_data, y_data

    if sample_fraction <= 0.0:
        raise ValueError('sample_fraction must be in the range (0, 1].')

    x_reset = x_data.reset_index(drop=True)
    y_reset = pd.Series(y_data).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    target_samples = max(1, int(round(len(y_reset) * sample_fraction)))
    class_counts = y_reset.value_counts()
    class_proportions = class_counts / len(y_reset)
    class_targets = (class_proportions * target_samples).round().astype(int).clip(lower=1)

    selected_indices = []
    y_values = y_reset.to_numpy()

    for class_value, target_size in class_targets.items():
        class_indices = np.flatnonzero(y_values == class_value)
        take = min(int(target_size), len(class_indices))

        if take > 0:
            selected_indices.append(rng.choice(class_indices, size=take, replace=False))

    if not selected_indices:
        return x_data, y_data

    sampled_indices = np.concatenate(selected_indices)

    if len(sampled_indices) > target_samples:
        sampled_indices = rng.choice(sampled_indices, size=target_samples, replace=False)
    elif len(sampled_indices) < target_samples:
        remaining = np.setdiff1d(
            np.arange(len(y_reset)),
            sampled_indices,
            assume_unique=False,
        )

        add_count = min(target_samples - len(sampled_indices), len(remaining))

        if add_count > 0:
            add_indices = rng.choice(remaining, size=add_count, replace=False)
            sampled_indices = np.concatenate([sampled_indices, add_indices])

    rng.shuffle(sampled_indices)

    return x_reset.iloc[sampled_indices], y_reset.iloc[sampled_indices]


def make_fixed_sampled_folds(folds, sampling: FoldSamplingConfig):
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


def bootstrap_stat_ci(scores, stat='mean', n_bootstrap=2000, alpha=0.05, seed=315):
    """Estimate a confidence interval for mean or median using bootstrap resampling."""

    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError('scores must contain at least one value.')

    if stat == 'mean':
        stat_fn = np.mean
    elif stat == 'median':
        stat_fn = np.median
    else:
        raise ValueError("stat must be either 'mean' or 'median'.")

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_stats[idx] = float(stat_fn(sample))

    lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        'value': float(stat_fn(values)),
        'ci_lower': lower,
        'ci_upper': upper,
    }


def summarize_with_ci(scores, n_bootstrap=2000, alpha=0.05, seed=315):
    """Return mean and median with bootstrap confidence intervals."""

    mean_summary = bootstrap_stat_ci(
        scores,
        stat='mean',
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    median_summary = bootstrap_stat_ci(
        scores,
        stat='median',
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed + 1,
    )

    return {
        'mean': mean_summary,
        'median': median_summary,
    }
