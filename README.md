# Bunq Ynab Connect

## Table of contents
- [Introduction](#introduction)
- [Installation & Configuration](#installation--configuration)
  - [Linking Ynab accounts with Bunq accounts](#linking-ynab-accounts-with-bunq-accounts)
- [Usage](#usage)
- [Testing](#testing)
- [Licence](#licence)
- [Contributing](#contributing)

## Introduction
This repository can be used to link Bunq to Ynab. Unfortunately, Ynab [does not support](https://syncforynab.com/banks) automatic for Bunq. This repository contains code and explanations on how to achieve this. Moreover, it contains code to train a model on your personal Bunq and Ynab data, to automatically classify transactions into your Ynab categories. It uses Docker and Docker compose to deploy your code, for example to a Raspberry Pi using Portainer.

## Installation & Configuration
The easiest way to get up and running, is using Docker compose. Moreover, to easily manage and monitor all container, you can use Portainer. To install Docker and Docker compose, follow the instructions on the [Docker website](https://docs.docker.com/get-docker/). To install Portainer, follow the instructions on the [Portainer website](https://www.portainer.io/installation/).

With portainer up and running, create a new stack, using [portainer.yaml](docker/portainer.yaml). Use [.docker-env.example](.docker-env.example) with values of your choice, and add them to the stack. Some information on some environment variables:

- `MONGO_URI` and `MONGO_DB` define your Mongo location and DB name. You probably don't need to change these.
- `MONGO_USER` and `MONGO_PASSWORD` define the user and password for your Mongo DB. You should change these to your liking.
- `BUNQ_ONETIME_TOKEN` is a token you can create in the Bunq app, see [Bunq API key](https://doc.bunq.com/#:~:text=Create%20an%20API%20key.,%E2%86%92%20Developers%20%E2%86%92%20API%20keys). This token is only used upon your first sync, to exchange it for a config file.
- `YNAB_TOKEN` is your [Ynab PAT](https://api.youneedabudget.com/#personal-access-tokens). 
- `START_SYNC_DATE` Defines what Bunq transactions will be synced to Ynab. Upon your first sync, your MongoDB is ingsted with all your Bunq payments. All payments that occured after the START_SYNC_DATE will be synced to Ynab. It is therefor important to set this value with care. Most likely, it is OK to set this to the current date. If you'd set it in in the past, all your bunq payments since then are synced to Ynab, while you've probably already entered them manually. Note that you can easily identify auto-synced payments in Ynab by the blue flag:large_blue_circle:.
- `PREFECT_*` define some urls for prefect. You probably don't need to change these. You can find more info on these in the prefect docs.
- `MLFLOW_TRACKING_URI` is the location of your MLFlow server. Again, probably no need to change this.
- `MLSERVER_URL` is the location of your MLServer.

Before you can start syncing, you must define what Ynab budgets and accounts belong to what Bunq accounts.

### Linking Ynab accounts with Bunq accounts
The link between these two is created using the description of your Ynab account. The project can simultaneously link and sync as many accounts of as many budgets with as many payment accounts as you like. For each account to link, open it in Ynab, and press the edit icon:pencil2:. Enter the IBAN you which to link into the "Account Notes" section. The code will use an exact match on this text field. 


## Usage
Prefect is used for orchestration and monitoring. This is therefor your best starting point to start syncing. Open it on [http://localhost:12002](http://localhost:12002) (or your servers' IP). You'll find the active deployments in the [deployments](http://localhost:12002/deployments) tab. Thw two most important deployments are:
- `sync`. It runs ourly, and syncs all unsynced transactions. It keeps track of everything in your MongoDB (which you can access at [http://localhost:12001](http://localhost:12001)), so rerunning the sync will not create duplicates.
- `train`. It runs weekly, and trains a classification model on your personal transactions and budget. 


## Testing
Pytest is used for testing. Unfortunately, the code coverage is not yet great. You can run the tests using Pytest, for example in VSCode.

## Technical documentation
Please find technical documentation in the [docs](docs) folder. Moreover, some subfolders contain markdown files with details on the classes and functions in that folder.

## Licence
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are most welcome. I've not yet setup a contributing guide, but feel free to reach out to me if you'd like to contribute to and/or use this project.

## Contact
Feel free to contact me at [aucke.bos97@gmail.com](mailto:aucke.bos97@gmail.com). 
