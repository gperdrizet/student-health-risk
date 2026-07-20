# Dev Container

Configuration for the VS Code dev container used for development.

## `devcontainer.json`

- **Base image**: `mcr.microsoft.com/devcontainers/python:3.12-trixie` - mirrors the Python version used in Kaggle notebooks.
- **Network**: uses bridge networking to ensure internet access from within the container. Port 8888 is bound to all host interfaces (`0.0.0.0`) so a Jupyter server started inside the container is accessible to other machines on the LAN.
- **LAN Jupyter server**: to serve notebooks to other devices on the network, start Jupyter with:
  ```
  jupyter notebook --ip=0.0.0.0 --no-browser
  ```
  Then open `http://<host-ip>:8888` in a browser on any LAN device, using the token printed in the terminal output.
- **On create**: upgrades `pip` and installs Python dependencies from `requirements.txt`.
- **Post create**: installs lightweight helper tools only. Large data files should be hosted on an external server and fetched separately.
- **VS Code extensions**: Python language support, Jupyter notebooks, and spell checking. ESLint is explicitly excluded as it is not needed for a Python project.
- **VS Code settings**: sets the default Python interpreter to `/usr/local/bin/python` and hides the system Python 3.13 environment to avoid kernel confusion in Jupyter.
