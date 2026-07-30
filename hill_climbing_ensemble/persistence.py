"""Persistence helpers for run artifacts and checkpoints."""

from __future__ import annotations

import json
import pickle
from pathlib import Path


def load_pickle(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def build_run_state(accepted_specs, hill_log, current_scores, current_summary, proposal_index):
    return {
        "accepted_specs": accepted_specs,
        "hill_log": hill_log,
        "current_scores": current_scores,
        "current_summary": current_summary,
        "last_proposal_index": proposal_index,
    }
