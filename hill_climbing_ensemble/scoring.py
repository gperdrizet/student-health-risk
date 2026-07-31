"""Scoring and prediction routines for hill-climbing ensembles."""

from __future__ import annotations

import pandas as pd

from sklearn.metrics import balanced_accuracy_score

from .ml_utils import (
    XGBoostError,
    average_probabilities,
    bootstrap_rows_and_features,
    build_xgb_model,
    make_fixed_sampled_folds,
)


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

    try:
        model.fit(x_bootstrap, y_bootstrap)
    except XGBoostError:
        model = build_xgb_model(spec["params"], seed=row_seed, prefer_gpu=False)
        model.fit(x_bootstrap, y_bootstrap)

    return model


def evaluate_ensemble_specs(folds, specs, sampling_config, seed):
    if not specs:
        return []

    evaluation_folds = make_fixed_sampled_folds(folds, sampling_config)
    fold_scores = []

    for fold_index, fold in enumerate(evaluation_folds, start=1):
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
        fold_scores.append(balanced_accuracy_score(fold["y_validation"], predictions))

    return fold_scores


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
