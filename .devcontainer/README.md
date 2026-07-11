# Dev Container

Configuration for the VS Code dev container used for development.

## `devcontainer.json`

- **Base image**: `mcr.microsoft.com/devcontainers/python:3.12-trixie` - mirrors the Python version used in Kaggle notebooks.
- **Network**: uses bridge networking to ensure internet access from within the container.
- **On create**: upgrades `pip` and installs Python dependencies from `requirements.txt`.
- **Post create**: installs Git LFS and configures it to track `*.csv` files so large data files are managed outside the main Git history.
- **VS Code extensions**: Python language support, Jupyter notebooks, and spell checking. ESLint is explicitly excluded as it is not needed for a Python project.
- **VS Code settings**: sets the default Python interpreter to `/usr/local/bin/python` and hides the system Python 3.13 environment to avoid kernel confusion in Jupyter.
