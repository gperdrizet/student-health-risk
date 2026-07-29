# Feature Engineering and Ensemble Plan

## Scope
- Leave notebooks 04 and earlier untouched.
- Add a new engineered-features notebook after notebook 04.
- Add a lightly tuned gradient boosting baseline on the engineered folds.
- Add new stacking notebooks that consume the engineered artifacts.

## Workflow
1. Load the preprocessed fold pickle and train/test CSVs produced by notebook 03.
2. Add leakage-safe engineered features using only row-wise transforms or fold-local fitting.
3. Save versioned engineered artifacts under `data/tmp/`.
4. Fit and score a compact gradient boosting baseline on the engineered folds.
5. Build a basic stacking ensemble on the engineered artifacts.
6. Optimize the stacking ensemble on the same engineered feature set.

## Feature engineering rules
- Do not modify notebook 04 or earlier.
- Do not fit transforms on the full dataset before splitting.
- Keep the first feature set small and deterministic.
- Use sampling for exploratory sweeps and reserve full-fold runs for shortlisted candidates.
- Favor engineered ratios, bins, and aggregate indicators over large polynomial expansions.

## Proposed artifacts
- `data/tmp/08-engineered-cv-folds.pkl`
- `data/tmp/08-engineered-train-data.csv`
- `data/tmp/08-engineered-test-data.csv`
- `data/results/09-gradient-boosting-scores.pkl`
- `data/results/10-stacking-cross-val-scores.pkl`
- `data/results/11-optimized-stacking-results.pkl`

## Notebook sequence
1. `08-feature-engineering.ipynb`
2. `09-gradient-boosting-baseline.ipynb`
3. `10-stacking-ensemble-engineered.ipynb`
4. `11-stacking-optimization-engineered.ipynb`

## Verification
- Confirm the engineered train/test artifacts share the same feature columns.
- Confirm no NaNs or infinities are introduced.
- Compare the gradient boosting baseline to the stacking ensemble.
- Compare the optimized stack to the baseline stack on the same engineered features.
- Re-run the engineering notebook once to check that output schema and row order are stable.