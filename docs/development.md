# Introduction
This page describes the development process of this project.

# Docker
To prevent the need of running different servers on your host (mlflow, mongo, etc), you should run the compose file on your host while developping. A specific compose file [development.yaml](docker/development.yaml) is available for this purpose. It mounts the code to the `prefect-agent` container, such that any changes you make are reflected in the container directly. This compose file uses the `.docker-env` file, as described in the [README](/README.md).

To be able to debug your code on your host, also create create a `.env` file, based on [.env.example](/.env.example). This file is used when you run your code on your host, and the example values make sure you use the running docker containers to store data, orchestrate and track experiments. To create a local python environment, run `rye sync` in the root of the project. You can now run your code locally, and debug it using VSCode. Any data is stored in your running containers, and you can use all the servers as in production. 

To debug code locally while using production data, you could update your `.env` file to reference the containers on the production server. For this to work, you must be in the same network. This can be used temporarily for debugging purposes.

# Code style
Ruff is used for code formatting. A step in the [cicd pipeline](/.github/workflows/cicd.yml) checks if the code is formatted correctly. You can run black using `ruff check`.

# Tasks and commands
Some VSCode tasks are defined in the project. Moreover, often-used commands are stored and described in [commands.md](/docs/commands.md).

## Dev Containers (VS Code)
An optional devcontainer is provided in `.devcontainer/` that lets you develop inside a container with Python 3.11 and Rye preinstalled while reusing the existing Docker Compose stack for MongoDB, MLflow, Prefect, etc.

- Open this repository in VS Code and run: "Dev Containers: Reopen in Container".
- The devcontainer will start alongside the services from `docker/development.yml`.
- Your workspace is mounted at `/workspaces/bunq-ynab-connect`. The Python environment is managed by Rye; dependencies are installed automatically with `rye sync` on first start.
- The container has Docker CLI access; you can run `docker compose -f docker/development.yml up` from the integrated terminal if you prefer starting services from within the container.

Ports forwarded: 12001 (mongo-express), 12002 (Prefect UI), 12003 (MLflow), 12005 (mlserver).

Benefits:
- Consistent Python toolchain (3.11, Rye, Ruff) across machines
- Clean host machine (no need to install Python toolchains locally)
- Works on Windows, macOS, and Linux the same way
