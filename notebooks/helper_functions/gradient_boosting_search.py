"""Search payload and analysis helpers for the gradient boosting baseline notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_stage1_payload(stage1_rows, best_params, config):
    """Create a stable payload shape for stage-1 search results."""

    return {
        'stage1_rows': stage1_rows,
        'rerank_rows': [],
        'best_params_stage1': best_params,
        'best_params_selected': best_params,
        'config': dict(config),
    }


def attach_rerank_results(search_payload, rerank_rows, best_params_selected, config_updates=None):
    """Attach optional rerank results without overwriting stage-1 results."""

    payload = dict(search_payload)
    payload['rerank_rows'] = list(rerank_rows)
    payload['best_params_selected'] = dict(best_params_selected)

    if config_updates:
        merged_config = dict(payload.get('config', {}))
        merged_config.update(config_updates)
        payload['config'] = merged_config

    return payload


def normalize_search_payload(raw_payload):
    """Normalize legacy list payloads to the stable dictionary schema."""

    if isinstance(raw_payload, list):
        stage1_rows = list(raw_payload)
        best_params = stage1_rows[0]['params']
        return build_stage1_payload(stage1_rows, best_params, config={})

    payload = dict(raw_payload)

    if 'stage1_rows' not in payload:
        raise KeyError('Expected search payload to contain stage1_rows.')

    payload.setdefault('rerank_rows', [])

    if 'best_params_stage1' not in payload:
        payload['best_params_stage1'] = payload.get('best_params', payload['stage1_rows'][0]['params'])

    if 'best_params_selected' not in payload:
        payload['best_params_selected'] = payload.get('best_params', payload['best_params_stage1'])

    payload.setdefault('config', {})
    return payload


def build_score_interval_frame(rows):
    """Return candidate medians and percentile intervals for plotting."""

    return pd.DataFrame({
        'median_score': [row['median'] for row in rows],
        '95% CI lower': [np.percentile(row['fold_scores'], 2.5) for row in rows],
        '95% CI upper': [np.percentile(row['fold_scores'], 97.5) for row in rows],
    })


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
