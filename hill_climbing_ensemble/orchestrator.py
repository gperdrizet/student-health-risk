"""Optuna-tracked hill-climbing ensemble orchestration."""

from __future__ import annotations

from copy import deepcopy
import logging
import pickle

import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .ml_utils import FoldSamplingConfig, summarize_scores, summarize_with_ci
from .candidate_generator import derive_parameter_ranges, draw_candidate_spec, limited_folds
from .config import HillClimbConfig
from .persistence import build_run_state, load_pickle, save_pickle
from .scoring import build_submission_from_specs, evaluate_ensemble_specs
from .submission_ops import auto_commit_tag_push, save_submission_csv, validate_submission_df


def _configure_logger(config: HillClimbConfig) -> logging.Logger:
    logger = logging.getLogger('hill_climbing_ensemble.run')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    if logger.handlers:
        logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(config.runtime_log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _load_artifacts(config: HillClimbConfig):
    with config.engineered_cv_folds.open("rb") as handle:
        engineered_folds = pickle.load(handle)

    with config.xgb_search_results.open("rb") as handle:
        search_payload = pickle.load(handle)

    train_df = pd.read_csv(config.engineered_train_data)
    test_df = pd.read_csv(config.engineered_test_data)

    raw_train = pd.read_csv(
        "https://media.githubusercontent.com/media/gperdrizet/fullstack-2605/"
        "refs/heads/main/data/student-health-risk-train.csv"
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(raw_train["health_condition"])

    return engineered_folds, search_payload, train_df, test_df, label_encoder


def _build_or_load_study(config: HillClimbConfig):
    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    return optuna.create_study(
        study_name=config.study_name,
        storage=config.optuna_storage_url,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )


def run_hill_climb(config: HillClimbConfig):
    config.ensure_parent_dirs()
    logger = _configure_logger(config)

    engineered_folds, search_payload, train_df, test_df, label_encoder = _load_artifacts(config)
    parameter_ranges = derive_parameter_ranges(search_payload)

    fast_folds = limited_folds(engineered_folds, config.fast_cv_fold_limit)
    fast_sampling = FoldSamplingConfig(
        use_sampling=config.fast_cv_use_sampling,
        train_sample_fraction=config.fast_cv_train_sample_frac,
        validation_sample_fraction=config.fast_cv_validation_sample_frac,
        sample_seed=config.random_seed + 100,
    )

    all_feature_columns = [column for column in train_df.columns if column != "health_condition"]

    run_state = load_pickle(config.run_state_file, default=None)
    if run_state:
        accepted_specs = run_state.get("accepted_specs", [])
        hill_log = run_state.get("hill_log", [])
        current_scores = run_state.get("current_scores", [])
        current_summary = run_state.get("current_summary", {"median": 0.0})
        start_index = int(run_state.get("last_proposal_index", 0)) + 1
    else:
        accepted_specs = []
        hill_log = []
        current_scores = []
        current_summary = {"median": 0.0}
        start_index = 1

    study = _build_or_load_study(config)

    logger.info(
        'run_start target_accepted=%d max_proposals=%d resume_proposal=%d study=%s storage=%s',
        config.target_accepted_models,
        config.max_proposals,
        start_index,
        config.study_name,
        config.optuna_storage_url,
    )

    for proposal_index in range(start_index, config.max_proposals + 1):
        trial = study.ask()

        candidate_spec = draw_candidate_spec(
            trial,
            parameter_ranges=parameter_ranges,
            all_feature_columns=all_feature_columns,
            random_seed=config.random_seed,
            row_fraction_range=config.model_row_fraction_range,
            feature_fraction_range=config.model_feature_fraction_range,
        )

        candidate_specs = accepted_specs + [candidate_spec]

        proposed_scores = evaluate_ensemble_specs(
            fast_folds,
            candidate_specs,
            sampling_config=fast_sampling,
            seed=config.random_seed + 500,
        )

        proposed_summary = summarize_scores(proposed_scores)
        current_median = float(current_summary["median"]) if accepted_specs else 0.0
        delta = proposed_summary["median"] - current_median
        accepted = delta > 0.0 or not accepted_specs

        if accepted:
            candidate_spec["weight"] = float(max(proposed_summary["median"], 1e-6))
            accepted_specs.append(candidate_spec)
            current_scores = proposed_scores
            current_summary = proposed_summary

        entry = {
            "proposal_index": proposal_index,
            "trial_number": trial.number,
            "accepted": accepted,
            "accepted_count": len(accepted_specs),
            "current_median_before": current_median,
            "proposed_median": float(proposed_summary["median"]),
            "delta": float(delta),
            "row_fraction": float(candidate_spec["row_fraction"]),
            "feature_fraction": float(candidate_spec["feature_fraction"]),
            "feature_count": len(candidate_spec["feature_columns"]),
            "params": deepcopy(candidate_spec["params"]),
        }
        hill_log.append(entry)

        trial.set_user_attr("accepted", accepted)
        trial.set_user_attr("accepted_count", len(accepted_specs))
        trial.set_user_attr("delta", float(delta))
        trial.set_user_attr("row_fraction", float(candidate_spec["row_fraction"]))
        trial.set_user_attr("feature_fraction", float(candidate_spec["feature_fraction"]))
        trial.set_user_attr("feature_count", len(candidate_spec["feature_columns"]))

        study.tell(trial, float(proposed_summary["median"]))

        params = candidate_spec['params']
        logger.info(
            'candidate proposal=%03d trial=%d accepted=%s accepted_count=%d delta=%.5f '
            'median_before=%.5f median_proposed=%.5f row_frac=%.3f feature_frac=%.3f feature_count=%d '
            'max_depth=%s learning_rate=%.6f n_estimators=%s',
            proposal_index,
            trial.number,
            accepted,
            len(accepted_specs),
            float(delta),
            float(current_median),
            float(proposed_summary['median']),
            float(candidate_spec['row_fraction']),
            float(candidate_spec['feature_fraction']),
            int(len(candidate_spec['feature_columns'])),
            params.get('max_depth'),
            float(params.get('learning_rate', float('nan'))),
            params.get('n_estimators'),
        )

        if accepted:
            logger.info(
                'retained proposal=%03d accepted_count=%d ensemble_mean=%.5f ensemble_median=%.5f ensemble_std=%.5f',
                proposal_index,
                len(accepted_specs),
                float(current_summary['mean']),
                float(current_summary['median']),
                float(current_summary['std']),
            )

        if accepted and len(accepted_specs) % config.checkpoint_every_accepts == 0:
            checkpoint_path = config.checkpoint_submission_dir / (
                f"08-xgb-ensemble-step-{len(accepted_specs):02d}.csv"
            )
            checkpoint_df = build_submission_from_specs(
                accepted_specs,
                train_df,
                test_df,
                label_encoder,
            )
            save_submission_csv(checkpoint_df, checkpoint_path)
            logger.info('checkpoint_saved path=%s', checkpoint_path)

        if accepted:
            submission_df = build_submission_from_specs(accepted_specs, train_df, test_df, label_encoder)
            validate_submission_df(submission_df)
            save_submission_csv(submission_df, config.final_submission_file)

        save_pickle(config.ensemble_log_file, hill_log)

        ensemble_payload = {
            "accepted_specs": accepted_specs,
            "fast_cv_summary": summarize_scores(current_scores) if current_scores else {},
            "config": {
                "target_accepted_models": config.target_accepted_models,
                "max_proposals": config.max_proposals,
                "fast_cv_fold_limit": config.fast_cv_fold_limit,
                "fast_cv_use_sampling": config.fast_cv_use_sampling,
                "fast_cv_train_sample_frac": config.fast_cv_train_sample_frac,
                "fast_cv_validation_sample_frac": config.fast_cv_validation_sample_frac,
                "model_row_fraction_range": config.model_row_fraction_range,
                "model_feature_fraction_range": config.model_feature_fraction_range,
                "random_seed": config.random_seed,
                "study_name": config.study_name,
            },
        }

        if config.run_final_cv_estimate and accepted_specs:
            final_folds = limited_folds(engineered_folds, config.final_cv_fold_limit)
            final_sampling = FoldSamplingConfig(
                use_sampling=config.final_cv_use_sampling,
                train_sample_fraction=config.final_cv_train_sample_frac,
                validation_sample_fraction=config.final_cv_validation_sample_frac,
                sample_seed=config.random_seed + 900,
            )
            final_scores = evaluate_ensemble_specs(
                final_folds,
                accepted_specs,
                sampling_config=final_sampling,
                seed=config.random_seed + 1200,
            )
            final_summary = summarize_scores(final_scores)
            final_summary["fold_count_used"] = len(final_folds)
            final_summary["fold_count_available"] = len(engineered_folds)
            final_summary["accepted_model_count"] = len(accepted_specs)
            final_summary["use_sampling"] = config.final_cv_use_sampling
            final_summary["mean_ci_95"] = summarize_with_ci(final_scores)["mean"]
            final_summary["median_ci_95"] = summarize_with_ci(final_scores)["median"]
            ensemble_payload["final_cv_summary"] = final_summary

        save_pickle(config.ensemble_model_file, ensemble_payload)

        run_state = build_run_state(
            accepted_specs=accepted_specs,
            hill_log=hill_log,
            current_scores=current_scores,
            current_summary=current_summary,
            proposal_index=proposal_index,
        )
        save_pickle(config.run_state_file, run_state)

        if accepted and config.git_automation.enabled:
            tag = auto_commit_tag_push(
                commit_paths=config.git_automation.commit_paths,
                lock_file=config.git_automation.lock_file,
                accepted_count=len(accepted_specs),
                median_score=current_summary["median"],
                proposal_index=proposal_index,
                dry_run=config.git_automation.dry_run,
            )
            logger.info('auto_submission_tag_pushed tag=%s', tag)

        if len(accepted_specs) >= config.target_accepted_models:
            logger.info('target_reached proposal=%d accepted_count=%d', proposal_index, len(accepted_specs))
            break

    logger.info(
        'run_complete accepted_models=%d total_proposals=%d final_median=%.5f',
        len(accepted_specs),
        len(hill_log),
        float(current_summary.get('median', 0.0)),
    )

    return {
        "accepted_models": len(accepted_specs),
        "total_proposals": len(hill_log),
        "current_summary": current_summary,
        "study_name": config.study_name,
        "storage": config.optuna_storage_url,
    }
