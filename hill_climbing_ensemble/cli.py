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
        "--gpu-ids",
        default="0,1",
        help="Comma-separated GPU IDs used in parallel for fold scoring (e.g. 0,1).",
    )
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
    parser.add_argument(
        "--sampler",
        default="tpe",
        choices=["tpe", "random"],
        help="Optuna sampler: 'tpe' (default) or 'random' for uniform exploration.",
    )
    parser.add_argument(
        "--wide-search",
        action="store_true",
        help="Replace notebook-07-derived parameter bounds with broad XGBoost-wide ranges.",
    )
    parser.add_argument(
        "--inherit-ensemble",
        action="store_true",
        help="Seed the new run from existing accepted models but reset the proposal counter.",
    )
    parser.add_argument(
        "--feature-fraction-range",
        default=None,
        help="Override ensemble feature bootstrap range as 'min,max' (e.g. 0.1,0.9).",
    )
    parser.add_argument(
        "--row-fraction-range",
        default=None,
        help="Override ensemble row bootstrap range as 'min,max' (e.g. 0.1,0.9).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gpu_ids = tuple(int(value.strip()) for value in args.gpu_ids.split(",") if value.strip())
    if not gpu_ids:
        raise ValueError("--gpu-ids must include at least one GPU id.")

    config = HillClimbConfig(
        study_name=args.study_name,
        max_proposals=args.max_proposals,
        target_accepted_models=args.target_accepted_models,
        checkpoint_every_accepts=args.checkpoint_every_accepts,
        random_seed=args.random_seed,
        parallel_gpu_ids=gpu_ids,
    )
    config.git_automation.enabled = not args.disable_auto_submit
    config.git_automation.dry_run = args.dry_run_git
    config.sampler_name = args.sampler
    config.wide_parameter_search = args.wide_search
    config.inherit_ensemble = args.inherit_ensemble

    if args.feature_fraction_range:
        lo, hi = (float(v.strip()) for v in args.feature_fraction_range.split(','))
        config.model_feature_fraction_range = (lo, hi)
    if args.row_fraction_range:
        lo, hi = (float(v.strip()) for v in args.row_fraction_range.split(','))
        config.model_row_fraction_range = (lo, hi)

    result = run_hill_climb(config)
    print("Run complete")
    print(result)


if __name__ == "__main__":
    main()
