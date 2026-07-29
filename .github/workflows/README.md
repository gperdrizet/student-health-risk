# GitHub Actions workflows

This directory manages the continuous integration and deployment pipeline for the Kaggle competition. The setup uses a localized terminal approach, completely eliminating the need for GitHub UI interaction.

## 1. `verify_submission.yml` - automated formatting verification

Runs instantly on every commit or update pushing directly to the `main` branch that alters the core submission file. It enforces strict compliance checks to protect daily Kaggle submission quotas against corrupted runs.

**Trigger**: Any `git push` directly to the `main` branch modifying `data/submission.csv`.

**Strict Data Constraints Enforced**:
* **Row Count**: Must contain exactly 295,753 lines of data (excluding the header).
* **Schema Integrity**: Must feature precisely two matching columns labeled `id` and `health_condition`.
* **Classification Bounds**: Predictions are strictly audited to contain only `'unhealthy'`, `'at-risk'`, or `'fit'`.
* **Index Boundaries**: Verifies test set integrity by asserting the first Row ID is exactly `690088` and the trailing Row ID matches `985840`.

---

## 2. `submit.yml` - release and Kaggle submission

Acts as the production pipeline delivery step. It runs on pushed version tags, creates a GitHub Release, uploads a version-stamped immutable copy of the submission file, submits that artifact to Kaggle, and then refreshes the README badge and leaderboard plot.

**Trigger**: Pushing a versioned semantic tag matching the `v*` pattern (e.g., `git push origin v0.1.0`).

**Execution Architecture Steps**:
1. Extracts the version tag from `GITHUB_REF`.
2. Generates release notes from [`.github/release.yml`](../release.yml) using commit-prefix grouping such as `model:` and `feature:`.
3. Creates a GitHub Release with the generated notes and uploads a versioned copy of `data/submission.csv`.
4. Submits the versioned artifact to Kaggle with the release notes as the submission message.
5. Waits for Kaggle scoring, fetches the leaderboard export, writes a score distribution plot, and updates the README badge and plot.

## Required authentication setup

The execution layer requires configuration of two sensitive variables within your repository repository actions portal (`Settings -> Secrets and variables -> Actions`):

* `KAGGLE_USERNAME`: Your personal Kaggle account username.
* `KAGGLE_API_TOKEN`: A valid API Token string generated via your Kaggle user profile settings page.

## Local workflow instructions

To deploy, simply commit changes locally, use semantic prefix messages, and push your version tags from your terminal:

```bash
# 1. Commit and push data adjustments to test layout automatically
git add .
git commit -m "model: upgraded stacking ensemble with a LightGBM meta-learner"
git push origin main

# 2. Tag your target structure once the layout verification step passes green
git tag v1.0.0
git push origin v1.0.0
```
