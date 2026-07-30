"""Candidate sampling utilities for hill-climbing ensemble proposals."""

from __future__ import annotations

import numpy as np


def limited_folds(folds, fold_limit):
    if fold_limit is None or fold_limit <= 0:
        return folds
    return folds[: min(fold_limit, len(folds))]


def select_feature_subset(feature_columns, feature_fraction, seed):
    if feature_fraction >= 1.0:
        return list(feature_columns)

    rng = np.random.default_rng(seed)
    total = len(feature_columns)
    target = max(1, int(round(total * feature_fraction)))
    return sorted(rng.choice(np.array(feature_columns), size=target, replace=False).tolist())


def derive_parameter_ranges(search_payload):
    ranges = search_payload.get("parameter_ranges")
    if ranges:
        return ranges

    rerank_rows = search_payload.get("rerank_rows", search_payload["stage1_rows"])
    top_rows = rerank_rows[: min(8, len(rerank_rows))]
    if not top_rows:
        raise ValueError("Unable to derive parameter ranges from empty search payload rows.")

    ranges = {}
    for key in top_rows[0]["params"].keys():
        values = [row["params"][key] for row in top_rows]
        if isinstance(values[0], int):
            ranges[key] = [int(min(values)), int(max(values))]
        else:
            ranges[key] = [float(min(values)), float(max(values))]

    return ranges


def draw_candidate_spec(trial, parameter_ranges, all_feature_columns, random_seed, row_fraction_range, feature_fraction_range):
    params = {}
    for key, value_range in parameter_ranges.items():
        low, high = value_range
        is_int = isinstance(low, int) and isinstance(high, int)
        if is_int:
            params[key] = int(trial.suggest_int(key, int(low), int(high)))
        else:
            params[key] = float(trial.suggest_float(key, float(low), float(high)))

    row_fraction = float(
        trial.suggest_float("row_fraction", float(row_fraction_range[0]), float(row_fraction_range[1]))
    )
    feature_fraction = float(
        trial.suggest_float(
            "feature_fraction",
            float(feature_fraction_range[0]),
            float(feature_fraction_range[1]),
        )
    )

    proposal_index = trial.number + 1
    feature_seed = random_seed + 10000 + proposal_index
    feature_columns = select_feature_subset(
        all_feature_columns,
        feature_fraction=feature_fraction,
        seed=feature_seed,
    )

    return {
        "params": params,
        "row_fraction": row_fraction,
        "feature_fraction": feature_fraction,
        "feature_columns": feature_columns,
        "feature_seed": feature_seed,
        "proposal_seed": random_seed + proposal_index,
    }
