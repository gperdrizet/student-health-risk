"""Helper functions for the engineered-feature gradient boosting baseline notebook."""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score


def build_model(params, seed=315):
    """Build a class-balanced HistGradientBoostingClassifier from params."""
    return HistGradientBoostingClassifier(
        **params,
        class_weight='balanced',
        random_state=seed,
    )


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
        remaining = np.setdiff1d(np.arange(len(y_reset)), sampled_indices, assume_unique=False)
        add_count = min(target_samples - len(sampled_indices), len(remaining))

        if add_count > 0:
            add_indices = rng.choice(remaining, size=add_count, replace=False)
            sampled_indices = np.concatenate([sampled_indices, add_indices])

    rng.shuffle(sampled_indices)
    return x_reset.iloc[sampled_indices], y_reset.iloc[sampled_indices]


def score_on_folds(
    folds,
    params,
    use_sampling=False,
    train_sample_fraction=None,
    validation_sample_fraction=None,
):
    """Evaluate one parameter set across CV folds using balanced accuracy."""
    fold_scores = []

    for fold_index, fold in enumerate(folds, start=1):
        model = build_model(params, seed=315 + fold_index)

        x_train = fold['x_train']
        y_train = fold['y_train']
        x_validation = fold['x_validation']
        y_validation = fold['y_validation']

        if use_sampling:
            x_train, y_train = sample_fold_split(
                x_train,
                y_train,
                sample_fraction=train_sample_fraction,
                seed=1000 + fold_index,
            )

            x_validation, y_validation = sample_fold_split(
                x_validation,
                y_validation,
                sample_fraction=validation_sample_fraction,
                seed=2000 + fold_index,
            )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_validation)
        fold_scores.append(balanced_accuracy_score(y_validation, y_pred))

    return fold_scores


def build_relative_candidate(base_params, spec):
    """Create one candidate by applying relative tweaks to base params."""
    candidate = dict(base_params)

    if 'learning_rate_mult' in spec and 'learning_rate' in base_params:
        candidate['learning_rate'] = max(
            1e-4,
            float(base_params['learning_rate']) * float(spec['learning_rate_mult']),
        )

    if 'max_iter_mult' in spec and 'max_iter' in base_params:
        candidate['max_iter'] = max(
            50,
            int(round(float(base_params['max_iter']) * float(spec['max_iter_mult']))),
        )

    if 'max_depth_delta' in spec and 'max_depth' in base_params:
        candidate['max_depth'] = max(
            2,
            int(round(float(base_params['max_depth']) + float(spec['max_depth_delta']))),
        )

    if 'max_features_mult' in spec and 'max_features' in base_params:
        candidate['max_features'] = float(base_params['max_features']) * float(spec['max_features_mult'])
        candidate['max_features'] = min(1.0, max(0.1, candidate['max_features']))

    return candidate
