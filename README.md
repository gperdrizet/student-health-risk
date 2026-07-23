# student-health-risk

[![Submit to Kaggle](https://github.com/gperdrizet/student-health-risk/actions/workflows/submit.yml/badge.svg)](https://github.com/gperdrizet/student-health-risk/actions/workflows/submit.yml)

Kaggle Playground Series Season 6 Episode 7 solution. In addition to the S6E7 solution, this repository provides a convenient containerized development environment for Kaggle competitions, and an automated submission workflow using GitHub actions.


## Notebooks

1. [`01-EDA.ipynb`](https://github.com/gperdrizet/student-health-risk/blob/main/notebooks/01-EDA.ipynb): Data exploration and analysis of features and labels
2. [`02-gradient-boosting.ipynb`](https://github.com/gperdrizet/student-health-risk/blob/main/notebooks/02-gradient-boosting.ipynb): Basic Scikit-learn gradient boosting solution with minimal data preprocessing.
3. [`03-data-preprocessing`](https://github.com/gperdrizet/student-health-risk/blob/main/notebooks/03-data-preprocessing.ipynb): Simple optimization of imputation and categorical feature encoding strategies, with Scikit-learn gradient boosting model trained on whole dataset for submission.


## Submissions

| Submission                 | Pull request | Estimated balanced accuracy | Leaderboard balanced accuracy | Leaderboard rank              |
|----------------------------|--------------|-----------------------------|-------------------------------|-------------------------------|
| 1. Majority class          | PR #10       | 33.3%                       | 33.3%                         | 1380                          |
| 2. Gradient boosting tree  | PR #12       | 86.6% - 87.2%               | 86.3%                         | 1184                          |
| 3. Optimized preprocessing | PR #15       | 86.9% - 87.7%               | 87.6%                         | 1855/2332 (79.5th percentile) |


## Submission workflow

1. Work on `dev` branch, commit & push changes to GitHub
2. When new submission is ready, open pull request against main
3. Merging pull request triggers GitHub actions workflow to submit `data/submission.csv`

**Note**: you must add `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME` to GitHub action secretes for Kaggle submission to work.


## Development environment

Development work is done in a Python 3.12 devcontainer which aims to recapitulate the Kaggle notebook environment. The environment and GitHub actions setup should be generally useful for Kaggle competitions. Please feel free to use this repository as a template for your own submissions:


### Prerequisites

1. Docker Desktop or Docker Engine
2. VS Code
3. git


### Steps

1. Fork and clone this repository
2. Open it in a devcontainer with VS Codes's `Dev Container: Open Folder in Container` command


## Helpful documentation

- [Kaggle CLI tutorial](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md)
- [GitHub workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [GitHub actions marketplace](https://github.com/marketplace?type=actions)


## Citation

If you use any part of this repository in your own work, please cite it:

```
@misc{student-health-risk,
  author = {gperdrizet},
  title  = {student-health-risk},
  year   = {2026},
  url    = {https://github.com/gperdrizet/student-health-risk}
}
```


## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).