# Introduction
All parts of the project run in Docker containers. This page lists the containers, and their purpose.

## Containers
The following containers are defined in the main compose files ([development.yml](../docker/development.yml), [portainer.yml](../docker/portainer.yaml)). `development.yml` should be used during development, see [development.md](./development.md). `portainer.yml` should be used in production, see [README.md](../README.md).
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
- Copy the required files, and install the pyproject.toml file. 
- Install required dependencies
- Run the entrypoint. This serves all flows, using the [serve method](https://docs-3.prefect.io/3.0/deploy/run-flows-in-local-processes). 

### Watchtower
`watchtower` is a container that watches for changes in the `prefect-agent` container. When a change is detected, it pulls the new image, and restarts the container. This is used for an easy deployment mechanism. The [deploy.yaml](/.github/workflows/deploy.yaml) workflow builds the container upon every merge, and pushes it to Docker Hub. The watchtower container then pulls the new image, and restarts the `prefect-agent` container. 

### MLFlow
`mlflow` is a container that hosts the MLFlow server. It is used to track experiments and register models. It is available at [http://localhost:12003](http://localhost:12003).

### MLServer
`mlserver` is a container that hosts the MLServer. It is used to serve models. It retrieves the Models from the `mlflow` container, and makes makes them available as REST endpoints to other containers in the network (ie the `prefect-agent`). Whenever a payment is to be synced, a request to `mlserver` is made to classify it with the correct model. Swagger docs are available at [http://localhost:12006/v2/docs#/](http://localhost:12006/v2/docs#/)

### MLServer restarter
A temporary solution to the following problem:
A deployment trains a new model for each budget weekly. If the new model has a better performance then the existing model, it is promoted to Production. MLServer needs to be restarted to load the new model. However, the `prefect-agent` container does not have access to the docker socket on the host, so it cannot restart the `mlserver` container. 
This container uses the docker cli to restart the `mlserver` container daily. 

### Callback server
`callback` allows near-realtime syncing of Bunq Payments to Ynab. It exposes a Fastapi server, and registers itself as a Callback with Bunq. Bunq will POST any payment made to this URL, and the server will sync it to YNAB directly. Without this container, payments are synced hourly, using the [sync flow](./orchestration.md#deployments).

## Traefik
The [Callback server](#callback-server) exposes a FastAPI API server. For Bunq callbacks to work, the following constraints must be met:
- The server must contain a valid SSL Certificate
- The URL must be the default URL for HTTPS: 443. 

Because of these constraints, a setup using a [Traefik reverse proxy](https://doc.traefik.io/traefik/getting-started/quick-start/) is chosen. This way, we delegate maintaining valid SSL certifications to traefik, and still support other services running on port 443 on the same host. 

There are three options for configuring this:
### 1. Do not use near-realtime syncing
This setup is optional. If you do not use it, payments will be synced once every hour. To configure it as such, simply remove the `callback` container from your compose file. 

### 2. You have an existing Traefik reverse proxy
If you are already using Traefik for network handling on your host, you can keep using it. To make sure your callback server correctly registers with your proxy, perform the following steps:
- Set environment variable `BUNQ_CALLBACK_HOST` to a hostname for which you can create valid certificates.
- Make sure your Traefik proxy has a certificate resolver configured for the above host. Save the resolver name in env var `RESOLVER_NAME`
- Make sure your proxy exposes a `websecure` entrypoint, on port 443. If you use another name, update the compose accordingly. 

When upping the compose, the `callback` container should now register itself with your existing proxy, and it should be available on the configured host, under port `443`. Test this by going to `https://{BUNQ_CALLBACK_HOST}/docs` on your host. You should see a swagger API documentation, container one POST endpoint `/payments`. 

### 3. You do not yet have a Traefik reverse proxy
A separate compose file is added to this repository: [traefik.yml](../docker/traefik.yml). It contains a single container, which is a Traefik reverse proxy. The container is configured to use [Let's Encrypt](https://letsencrypt.org/) to create free SSL Certificates, and assumes you use [Cloudflare](https://www.cloudflare.com/) to manager your DNS records. If you wish to change this, update the compose file accordingly. 

To use this container, perform the following steps:
- Set environment variable `BUNQ_CALLBACK_HOST` to a hostname for which you can create valid certificates.
- Set environment variable `RESOLVER_NAME` to a resolver name of your liking. You could simply use `myresolver`.
- Set environment variable `LETSENCRYPT_EMAIL` to your email adres. Let's Encrypt will contact you on this email adres when your certificate is about to expire. 
- Set environment variable `CF_DNS_API_TOKEN` to your Cloudflare global API token. Traefik will use it to update your SSL Certificates. 
- Up the compose file. You can use the VSCode Task `[Traefik up]` for this. 

When upping the main compose, the `callback` container should now register itself with your proxy, and it should be available on the configured host, under port `443`. Test this by going to `https://{BUNQ_CALLBACK_HOST}/docs` on your host. You should see a swagger API documentation, container one POST endpoint `/payments`. 