# GitHub Actions Workflows

## `submit.yml` - Submit to Kaggle

Automates submission of `data/submission.csv` to the Kaggle competition whenever a pull request is merged into `main`. Can also be triggered manually via `workflow_dispatch`.

**Trigger**: `pull_request` closed against `main` (merged only) or manual dispatch.

**Steps**:
1. Checks out the repository. Data files are expected to be available outside Git, such as on an external server.
2. Sets up Python 3.12.
3. Installs the Kaggle CLI.
4. Submits `data/submission.csv` to the `playground-series-s6e7` competition, tagging the submission with the originating PR number.

**Required secrets**: `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` must be set in the repository's GitHub Actions secrets.
