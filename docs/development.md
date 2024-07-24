# Introduction
This page describes the development process of this project.

# Docker
To prevent the need of running different servers on your host (mlflow, mongo, etc), you should run the compose file on your host while developping. A specific compose file [development.yaml](docker/development.yaml) is available for this purpose. It mounts the code to the `prefect-agent` container, such that any changes you make are reflected in the container directly. This compose file uses the `.docker-env` file, as described in the [README](/README.md).

To be able to debug your code on your host, also create create a `.env` file, based on [.env.example](/.env.example). This file is used when you run your code on your host, and the example values make sure you use the running docker containers to store data, orchestrate and track experiments. To create a local python environment, run `poetry install` in the root of the project. You can now run your code locally, and debug it using VSCode. Any data is stored in your running containers, and you can use all the servers as in production. 

To debug code locally while using production data, you could update your `.env` file to reference the containers on the production server. For this to work, you must be in the same network. This can be used temporarily for debugging purposes.

# Code style
Black is used for code formatting. A Github action in the [build pipeline](/.github/workflows/build.yml) checks if the code is formatted correctly. You can run black using `poetry run black .`. There is currently a bug where the black autoformatter does not sort packages. This will result in a failing buid pipeline. A VSCode task is created to manually sort all imports.

# Tasks and commands
Some VSCode tasks are defined in the project. Moreover, often-used commands are stored and described in [commands.md](/docs/commands.md).
