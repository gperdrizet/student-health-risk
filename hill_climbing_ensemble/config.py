"""Configuration objects for hill-climbing ensemble runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GitAutomationConfig:
    """Controls auto commit/tag/push behavior on accepted improvements."""

    enabled: bool = True
    dry_run: bool = False
    lock_file: Path = Path("data/results/08-hillclimb-git.lock")
    commit_paths: tuple[str, ...] = (
        "data/submission.csv",
        "data/results/08-xgboost-hillclimb-log.pkl",
        "data/results/08-xgboost-ensemble.pkl",
        "data/results/08-xgboost-hillclimb-run-state.pkl",
        "data/results/08-xgboost-hillclimb-optuna.db",
    )


@dataclass
class HillClimbConfig:
    """End-to-end configuration for the hill-climbing runtime."""

    engineered_cv_folds: Path = Path("data/tmp/05-engineered-cv-folds.pkl")
    engineered_train_data: Path = Path("data/tmp/05-engineered-train-data.csv")
    engineered_test_data: Path = Path("data/tmp/05-engineered-test-data.csv")
    xgb_search_results: Path = Path("data/results/07-xgboost-search-results.pkl")

    ensemble_log_file: Path = Path("data/results/08-xgboost-hillclimb-log.pkl")
    ensemble_model_file: Path = Path("data/results/08-xgboost-ensemble.pkl")
    run_state_file: Path = Path("data/results/08-xgboost-hillclimb-run-state.pkl")
    optuna_storage_file: Path = Path("data/results/08-xgboost-hillclimb-optuna.db")
    runtime_log_file: Path = Path("logs/08-xgboost-hillclimb-runtime.log")

    final_submission_file: Path = Path("data/submission.csv")
    checkpoint_submission_dir: Path = Path("data/submissions")

    target_accepted_models: int = 24
    max_proposals: int = 240
    checkpoint_every_accepts: int = 4

    fast_cv_fold_limit: int | None = 4
    fast_cv_use_sampling: bool = True
    fast_cv_train_sample_frac: float = 0.3
    fast_cv_validation_sample_frac: float = 0.3
    parallel_gpu_ids: tuple[int, ...] = (0, 1)

    final_cv_fold_limit: int | None = None
    final_cv_use_sampling: bool = False
    final_cv_train_sample_frac: float = 1.0
    final_cv_validation_sample_frac: float = 1.0

    model_row_fraction_range: tuple[float, float] = (0.60, 0.90)
    model_feature_fraction_range: tuple[float, float] = (0.25, 0.65)

    random_seed: int = 315
    study_name: str = "hillclimb_08"
    run_final_cv_estimate: bool = True

    git_automation: GitAutomationConfig = field(default_factory=GitAutomationConfig)

    @property
    def optuna_storage_url(self) -> str:
        return f"sqlite:///{self.optuna_storage_file}"

    def ensure_parent_dirs(self) -> None:
        for path in (
            self.ensemble_log_file,
            self.ensemble_model_file,
            self.run_state_file,
            self.optuna_storage_file,
            self.runtime_log_file,
            self.final_submission_file,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_submission_dir.mkdir(parents=True, exist_ok=True)
        self.git_automation.lock_file.parent.mkdir(parents=True, exist_ok=True)
