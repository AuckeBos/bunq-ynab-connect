# Introduction
All parts of the project run in Docker containers. This page lists the containers, and their purpose.

## Containers
### MongoDB
`mongo` is a MongoDB container. All data is stored in this container. It contains one database, named `bunq_ynab_connect. This stores the following collections:
- `bunq_accounts`: The list of all accounts in your Bunq account.
- `bunq_payments`: All payments in your Bunq account. 
- `matched_transactions`: A dataset of `ynab_transactions` <> `bunq_payments` transactions. Items are matched based on date and amount. The dataset is used as a train and test set for the classification model.
- `payment_clasiifications`: When a new payment is ingested, it is classified into a category in your budget. The classification is made by the model for this budget that is currently in production. The classification is stored, for debugging purposes and drift detection (todo).
- `payment_queue`: The processes of ingesting a payment and syncing it to Ynab are split into two steps. After ingestion, it is added to the queue. The queue is later processed by the sync process. This table stores the queue. It serves as a track record to see what has been synced, and what has not.
- `runmoments`: Is a delta table. It stores runmoments of several processes, to be able to load delta's instead of full loads. A [data extractor](/bunq_ynab_connect/data/data_extractors/) could use the last run moment to only extract data that has been added since the last run.
- `ynab_accounts`: The list of all accounts in your Ynab budget.
- `ynab_budgets`: The list of all budgets in your Ynab account.
- `ynab_transactions`: All transactions in your Ynab account. 

### Mongo Express
`mongo_express` is a web-based MongoDB admin interface. It is used to view and edit the data in the `mongo` container. It is available at [http://localhost:12001](http://localhost:12001).

### Prefect server
`prefect-server` is the Prefect server. It is used for orchestration and monitoring. It is available at [http://localhost:12002](http://localhost:12002). This container only hosts the server, and does not run any code (the code is not even mounted). 

### Prefect agent
`prefect-agent` is the Prefect agent. It uses a custom Dockerfile for the image. This container contains the code in this package. Upon boot, it deploys all the deployments to the server, and starts a worker, see [entrypoint.sh](/docker/entrypoint.sh).
The Dockerfile is described by:
- Python as a base image
- Setting up a user, folders and permissions
- Installing some required packages using apt-get
- Install poetry
- Copy the required files, and install the pyproject.toml file. 
- Run the entrypoint. It deploys all deployments in [prefect.yaml](/bunq_ynab_connect/prefect.yaml) to the server, and starts a worker.

### Watchtower
`watchtower` is a container that watches for changes in the `prefect-agent` container. When a change is detected, it pulls the new image, and restarts the container. This is used for an easy deployment mechanism. The [deploy.yaml](/.github/workflows/deploy.yaml) workflow builds the container upon every merge, and pushes it to Docker Hub. The watchtower container then pulls the new image, and restarts the `prefect-agent` container. 

### MLFlow
`mlflow` is a container that hosts the MLFlow server. It is used to track experiments and register models. It is available at [http://localhost:12003](http://localhost:12003).

### MLServer
`mlserver` is a container that hosts the MLServer. It is used to serve models. It retrieves the Models from the `mlflow` container, and makes makes them available as REST endpoints to other containers in the network (ie the `prefect-agent`). Whenever a payment is to be synced, a request to `mlserver` is made to classify it with the correct model. Swagger docs are available at [http://192.168.0.4:12006/v2/docs#/](http://192.168.0.4:12006/v2/docs#/)

### MLServer restarter
A temporary solution to the following problem:
A deployment trains a new model for each budget weekly. If the new model has a better performance then the existing model, it is promoted to Production. MLServer needs to be restarted to load the new model. However, the `prefect-agent` container does not have access to the docker socket on the host, so it cannot restart the `mlserver` container. 
This container uses the docker cli to restart the `mlserver` container daily. 
