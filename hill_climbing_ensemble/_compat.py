"""Compatibility imports for reusing notebook helper modules from a root package."""

from __future__ import annotations

from pathlib import Path
import sys

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from helper_functions.gradient_boosting_search import summarize_with_ci  # noqa: E402
from helper_functions.xgboost_ensemble import (  # noqa: E402
    FoldSamplingConfig,
    XGBoostError,
    average_probabilities,
    bootstrap_rows_and_features,
    build_xgb_model,
    make_fixed_sampled_folds,
    summarize_scores,
)
