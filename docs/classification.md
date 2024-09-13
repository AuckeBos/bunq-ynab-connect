# Introduction
Classification models are used to classify each Bunq transaction into one of the Categories to which the bunq account belongs. Because each budget has its own categories, a separate model is trained for each budget. The dataset for training and valiation is built by loading historical bunq and ynab transactions. These are then matched on amount + date. This match is not perfect, but results in a large and correct enough dataset to be usable. 

# Feature store
A features store is used to store the training data, and keep it up to date. The first step in the [train flow](/docs/orchestration.md#deployments) is to update the feature store. When the feature store is updated, it updates all datasets that belong to the feature store (currently only the [MatchedTransactionDataset](/bunq_ynab_connect/classification/datasets/matched_transactions_dataset.py)).

# Experiments
The models are created by means of experiments. Each experiment has a separate goal, but always includes training one or more models on a MatchedTransactionDataset. The [BasePaymentClassificationExperiment](/bunq_ynab_connect/classification/experiments/base_payment_classification_experiment.py) exposes functionality used by all experiments, like logging, loading the data, and train-test splitting.

## 1. Classifier selection experiment
This experiment is the first step in training the best model for a budget. Several classifiers are trained, with some default configuration. The model with the best cohens_kappa score is stored. 

## 2. Classifier tuning experiment
The second experiment to find the best model. It does a grid search on a grid of parameters, for the model found in experiment 1. The parameter combination with the best cohens_kappa score is stored.

## 3. Full training experiment
The full training experiment is the last step in training the best model. It trains the model with the best parameters found in experiment 2, on the full dataset. The goal is to store a model that is trained on as much data as is available. The best model is stored in MLFlow, as a [DeployableMlflowModel](/bunq_ynab_connect/classification/deployable_mlflow_model.py). This is a wrapper around the Sklearn model, to make sure predictions are logged to the `payment_classifications` table. This table will be used later for drift detection / performance monitoring.

# Training & Deployment
During the `train` flow, a [Trainer](/bunq_ynab_connect/classification/trainer.py) is instantiated for each budget. The training orchestrates all experiments. When finished, the [Deployer](/bunq_ynab_connect/classification/deployer.py) is used to deploy the model. This is only done if the cohens_kappa on the test set of the Full Training Experiment is better then the current model for this budget in production. Deployment hence means:
- Find the existing model for this budget in MLFlow. As mentioned, this model will always be a `DeployableMlflowModel`, which has the required label encoder and logging mechanisms in place.
- Check if the new model is better. Do not transtition it. The new version is stored, but not promoted.
- Else: Archive the old model, transition the new model to production.
- Create a config file for MLServer. This config file links to the Production stage of the model for this budget. Hence the URL will not change after first creation.
- `[Currently not working]` Restart MLServer. MLServer loads the models into memory, therefor if a new model is in the production stage, MLServer will not pick it up until it is restarted. Because the `prefect-agent` container has no access to the Docker CLI on the host, restarting the `mlserver` container fails. This is currently resolved by the [MLServer restarter](/docs/infrastructure.md#mlserver-restarter), which restarts the MLServer container daily.