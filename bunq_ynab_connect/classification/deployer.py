import json
from logging import LoggerAdapter

from kink import inject
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.tracking import MlflowClient

import mlflow
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.config import MLSERVER_CONFIG_DIR


class Deployer:
    """
    Deploy the model for a run. The run should be a run of the FullTrainingExperiment,
    which logs and registers the model.
    """

    budget_id: str
    storage: AbstractStorage
    logger: LoggerAdapter
    client: MlflowClient

    @inject
    def __init__(self, budget_id: str, storage: AbstractStorage, logger: LoggerAdapter):
        self.budget_id = budget_id
        self.storage = storage
        self.logger = logger
        self.client = MlflowClient()

    def deploy(self, run_id: str):
        """
        Deploy the model for the run.
        """

        models = mlflow.search_model_versions(
            filter_string=f"run_id='{run_id}' and name='{self.budget_id}'",
            max_results=1,
        )
        if len(models) == 0:
            raise ValueError(f"No registered model found for run {run_id}")
        model: ModelVersion = models[0]
        if self.run_is_better(model):
            self.transition_model(model)
            self.create_mlserver_config(model)
            self.logger.info("Model deployed")
        else:
            self.logger.info("Model not deployed")

    def create_mlserver_config(self, model: ModelVersion):
        run = mlflow.get_run(model.run_id)
        artifact_uri = run.info.artifact_uri
        model_uri = artifact_uri + "/deployable_model"

        data = {
            "name": self.budget_id,
            "implementation": "mlserver_mlflow.MlflowRuntime",
            "parameters": {"uri": model_uri},
        }
        destination = MLSERVER_CONFIG_DIR / f"{self.budget_id}.json"
        with open(destination, "w") as f:
            json.dump(data, f)
        self.logger.info(f"Created mlserver config at {destination}")

    def transition_model(self, model: ModelVersion):
        """
        Transition the model to production.
        """
        self.client.transition_model_version_stage(
            name=model.name,
            version=model.version,
            stage="Production",
            archive_existing_versions=True,
        )

    def run_is_better(self, new_model: ModelVersion) -> bool:
        """
        Check if the run is better than the current model.
        - If there is no current model, return True
        - Load the score of the current model
        - Load the score of the new run
        - If the new run is better, return True, else False
        """
        # Existing model
        existing_model_versions = self.client.get_latest_versions(
            self.budget_id, stages=["Production"]
        )
        if not existing_model_versions:
            self.logger.info("No existing model found")
            return True
        existing_model: ModelVersion = existing_model_versions[0]
        existing_run = mlflow.get_run(existing_model.run_id)
        existing_score = existing_run.data.metrics["cohen_kappa"]

        # New run
        new_run = mlflow.get_run(new_model.run_id)
        new_score = new_run.data.metrics["cohen_kappa"]
        # todo: >= to >
        if new_score >= existing_score:
            self.logger.info(f"New model is better: {new_score} > {existing_score}")
            return True
        else:
            self.logger.info(f"New model is worse: {new_score} <= {existing_score}")
            return False
