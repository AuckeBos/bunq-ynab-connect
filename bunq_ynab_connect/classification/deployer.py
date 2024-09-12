import json
from functools import cached_property
from logging import LoggerAdapter

from kink import inject

import docker
import mlflow
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.config import MLSERVER_CONFIG_DIR
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


class Deployer:
    """Deploy the model for a run.

    Run should be a run of FullTrainingExperiment, which logs and registers the model.
    """

    budget_id: str
    storage: AbstractStorage
    logger: LoggerAdapter
    client: MlflowClient
    PRODUCTION_ALIAS = "production"

    @inject
    def __init__(self, budget_id: str, storage: AbstractStorage, logger: LoggerAdapter):
        self.budget_id = budget_id
        self.storage = storage
        self.logger = logger
        self.client = MlflowClient()

    def deploy(self, run_id: str) -> None:
        """Deploy the model for the run, if the run is better than thecurrent model.

        If the run is better:
        - Transition the model to production
        - Create the mlserver config
        - Restart the mlserver service (in docker)
        """
        models = mlflow.search_model_versions(
            filter_string=f"run_id='{run_id}' and name='{self.budget_id}'",
            max_results=1,
        )
        if len(models) == 0:
            msg = f"No registered model found for run {run_id}"
            raise ValueError(msg)
        model: ModelVersion = models[0]
        if self.run_is_better(model):
            self.transition_model(model)
            self.create_mlserver_config()
            self.restart_mlserver()
            self.logger.info("Model deployed")
        else:
            self.logger.info("Model not deployed")

    def restart_mlserver(self) -> None:
        """Restart the mlserver service.

        Currently not working. When running inside an agent, it seems
        we have no access to the docker socket, hence the restart fails.
        Added restart_container to compose, to restart mlserver container daily
        """
        try:
            client = docker.DockerClient(base_url="unix://var/run/docker.sock")
            if container := client.containers.get("bunqynab_mlserver"):
                container.restart()
                self.logger.info("Restarted mlserver")
            else:
                self.logger.info("No mlserver container found, mlserver not restarted")
        except Exception:
            self.logger.exception("Error restarting mlserver")

    def create_mlserver_config(self) -> None:
        """Create the config file for mlserver, such that it can serve the model."""
        data = {
            "name": self.budget_id,
            "implementation": "mlserver_mlflow.MLflowRuntime",
            "parameters": {"uri": self.model_uri(self.budget_id)},
        }
        dir_ = MLSERVER_CONFIG_DIR / self.budget_id
        dir_.mkdir(exist_ok=True)
        destination = dir_ / "model-settings.json"
        with destination.open("w") as f:
            json.dump(data, f)
        self.logger.info("Created mlserver config at %s", destination)

    def model_uri(self, budget_id: str) -> str:
        return f"models:/{budget_id}@{self.PRODUCTION_ALIAS}"

    def transition_model(self, model: ModelVersion) -> None:
        """Transition the model to production."""
        self.client.set_registered_model_alias(
            name=model.name,
            alias=self.PRODUCTION_ALIAS,
            version=model.version,
        )

    def run_is_better(self, new_model: ModelVersion) -> bool:
        """Check if the run is better than the current model.

        - If there is no current model, return True
        - Load the score of the current model
        - Load the score of the new run
        - If the new run is better, return True, else False
        """
        # Existing model
        if not self.existing_model:
            return True
        existing_run = mlflow.get_run(self.existing_model.run_id)
        existing_score = existing_run.data.metrics["cohen_kappa"]

        # New run
        new_run = mlflow.get_run(new_model.run_id)
        new_score = new_run.data.metrics["cohen_kappa"]
        if new_score > existing_score:
            self.logger.info("New model is better: %s > %s", new_score, existing_score)
            return True
        self.logger.info("New model is worse: %s <= %s", new_score, existing_score)
        return False

    @cached_property
    def existing_model(self) -> ModelVersion | None:
        """Get the existing model."""
        try:
            return self.client.get_model_version_by_alias(
                name=self.budget_id, alias=self.PRODUCTION_ALIAS
            )

        except RestException:
            self.logger.exception("Could not load existing model")
            return None
