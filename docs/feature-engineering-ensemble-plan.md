# Feature Engineering and XGBoost Ensemble Plan

## Scope
- Keep notebooks 04 and earlier untouched.
- Expand feature diversity in notebook 05 and regenerate engineered artifacts.
- Keep notebook 06 as the refreshed single-model HGB benchmark.
- Add notebook 07 for optimized single-model XGBoost on engineered features.
- Add notebook 08 for all-XGBoost hill-climbing ensemble construction.

## Environment notes
- Devcontainer is being migrated to a CUDA-enabled Kaggle NVIDIA image.
- Notebook code should be GPU-first for XGBoost, but include CPU fallback for portability and debugging.
- Use deterministic seeds for fold sampling and candidate generation so CPU/GPU comparisons remain reproducible.

## Workflow
1. Rebuild engineered artifacts from notebook 05 after adding more diverse leakage-safe features.
2. Re-run notebook 06 to refresh the HGB benchmark and submission baseline.
3. Run notebook 07 sampled optimization to identify high-performing XGBoost regions quickly.
4. Use notebook 07 top candidates to define ranges for notebook 08 base learners.
5. Run notebook 08 hill climbing to build a fixed-size weighted ensemble and emit checkpoint submissions.

## Sampling design
- Fold-level fast CV sampling:
	- Used in notebook 07 optimization search and notebook 08 rapid ensemble evaluation.
	- Sample rows only from each fold split (no feature sampling at this level).
	- Keep the sampled fold subset and sampled row indices fixed for the duration of a run.
- Per-model diversity sampling:
	- Used in notebook 08 candidate training.
	- Bootstrap sample rows and sample feature subsets per model.
	- Track row seed, feature seed, and selected feature list per accepted model.

## Notebook contracts
1. `07-xgboost-baseline-engineered.ipynb`
	 - Medium-budget sampled search on limited folds.
	 - Full-fold rerank of top candidates.
	 - Cross-validation performance estimate with independent fold/sampling controls.
	 - Persist: stage-1 rows, rerank rows, best params, parameter ranges, CV summary.
2. `08-xgboost-hillclimb-ensemble.ipynb`
	 - Weighted-average hill climbing with fixed final size of 24 accepted models.
	 - Base learners are all XGBoost and differ by hyperparameters, row bootstrap, and feature subset.
	 - Keep/reject each candidate based on fast CV delta against current ensemble.
	 - Persist: full hill-climb log, accepted model registry, checkpoints, and final submission.

## Artifacts
- Inputs:
	- `data/tmp/05-engineered-cv-folds.pkl`
	- `data/tmp/05-engineered-train-data.csv`
	- `data/tmp/05-engineered-test-data.csv`
	- `data/results/06-gradient-boosting-scores.pkl`
- Notebook 07 outputs:
	- `data/results/07-xgboost-search-results.pkl`
	- `data/results/07-xgboost-scores.pkl`
- Notebook 08 outputs:
	- `data/results/08-xgboost-hillclimb-log.pkl`
	- `data/results/08-xgboost-ensemble.pkl`
	- `data/submission.csv` (best-known current submission)
	- Optional checkpoints: `data/submissions/08-xgb-ensemble-step-<n>.csv`

## Feature engineering rules
- No leakage: fit-only-on-train when transforms require fitting.
- Preserve stable feature ordering across fold/train/test outputs.
- Allow broad diversity, but keep each feature family deterministic and auditable.
- Validate no NaN/inf drift after transforms.

## Verification
- Schema parity:
	- engineered train/test columns match (except target/id roles).
	- each fold has identical `x_train` and `x_validation` feature columns/order.
- Reproducibility:
	- fixed sampled folds and row indices reproduce identical candidate rankings.
- Performance checks:
	- notebook 07 beats or matches notebook 06 under aligned CV protocol.
	- notebook 08 final weighted ensemble beats notebook 07 single-model median BA.
- Artifact checks:
	- all planned result files are written and contain run config metadata.