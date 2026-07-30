"""Command-line entrypoint for running hill-climbing ensemble search."""

from __future__ import annotations

import argparse

from .config import HillClimbConfig
from .orchestrator import run_hill_climb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Optuna-tracked hill-climbing ensemble search.")

    parser.add_argument("--study-name", default="hillclimb_08", help="Optuna study name.")
    parser.add_argument("--max-proposals", type=int, default=240, help="Max number of proposals to evaluate.")
    parser.add_argument(
        "--target-accepted-models",
        type=int,
        default=24,
        help="Stop after this many accepted ensemble members.",
    )
    parser.add_argument(
        "--checkpoint-every-accepts",
        type=int,
        default=4,
        help="Write a checkpoint submission every N accepted models.",
    )
    parser.add_argument("--random-seed", type=int, default=315)
    parser.add_argument(
        "--disable-auto-submit",
        action="store_true",
        help="Disable git tag/push automation on accepted improvements.",
    )
    parser.add_argument(
        "--dry-run-git",
        action="store_true",
        help="Print git operations without mutating repository state.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = HillClimbConfig(
        study_name=args.study_name,
        max_proposals=args.max_proposals,
        target_accepted_models=args.target_accepted_models,
        checkpoint_every_accepts=args.checkpoint_every_accepts,
        random_seed=args.random_seed,
    )
    config.git_automation.enabled = not args.disable_auto_submit
    config.git_automation.dry_run = args.dry_run_git

    result = run_hill_climb(config)
    print("Run complete")
    print(result)


if __name__ == "__main__":
    main()
