import json
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

    The run should have logged and registered a model of type DeployableMlflowModel.

    Attributes
    ----------
        budget_id: The budget id for the model
        storage: The storage to use
        logger: The logger to use
        client: The mlflow client to use
        PRODUCTION_ALIAS: The alias to use for production
        mlserver_repository_url: The url to the mlserver repository.
            Contains budget_id stub
        SCORE_DECREASE_THRESHOLD: The threshold for the score decrease to deploy
            If the score of the new model is less than this threshold worse than the
            existing model, the model will not be deployed. We allow for this margin,
            because we always prefer to deploy. This is because a new run includes
            all current categories of the budget.

    """

    budget_id: str
    storage: AbstractStorage
    logger: LoggerAdapter
    client: MlflowClient
    PRODUCTION_ALIAS = "production"
    mlserver_repository_url: str

    SCORE_DECREASE_THRESHOLD = 0.025

    @inject
    def __init__(
        self,
        budget_id: str,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        mlserver_repository_url: str,
    ):
        self.budget_id = budget_id
        self.storage = storage
        self.logger = logger
        self.client = MlflowClient()
        self.mlserver_repository_url = mlserver_repository_url

    def deploy(self, run_id: str) -> None:
        """Deploy the model for the run, if the run is better than thecurrent model.

        If the run is better:
        - Transition the model to production
        - Create the mlserver config
        - Restart the mlserver service (in docker)
        """
        new_model = self.new_model(run_id)
        if self.is_good_enough_to_deploy(new_model):
            self.transition_model(new_model)
            self.create_mlserver_config()
            self.restart_mlserver()
            self.logger.info("Model deployed")
        else:
            self.logger.info("Model not deployed")

    def new_model(self, run_id: str) -> ModelVersion:
        models = mlflow.search_model_versions(
            filter_string=f"run_id='{run_id}' and name='{self.budget_id}'",
            max_results=1,
        )
        if len(models) == 0:
            msg = f"No registered model found for run {run_id}"
            raise ValueError(msg)
        return models[0]

    @property
    def existing_model(self) -> ModelVersion | None:
        """Get the existing model."""
        try:
            return self.client.get_model_version_by_alias(
                name=self.budget_id, alias=self.PRODUCTION_ALIAS
            )

        except RestException:
            self.logger.exception("Could not load existing model", exc_info=False)  # noqa: LOG007
            return None

    def is_good_enough_to_deploy(self, new_model: ModelVersion) -> bool:
        if not (existing_model := self.existing_model):
            return True
        new_score = self.score(new_model)
        existing_score = self.score(existing_model)
        if (
            new_score > existing_score
            or abs(new_score - existing_score) < self.SCORE_DECREASE_THRESHOLD
        ):
            self.logger.info(
                "New score %s is good enough; old score was %s",
                new_score,
                existing_score,
            )
            return True
        self.logger.warning(
            "New score %s is not good enough; old score was %s",
            new_score,
            existing_score,
        )
        return False

    def score(self, model: ModelVersion) -> float:
        """Get the final score of the model."""
        run = mlflow.get_run(model.run_id)
        try:
            return run.data.metrics["final_score"]
        except KeyError:
            self.logger.warning("No final score found for run %s", model.run_id)
            return 0.0

    def transition_model(self, model: ModelVersion) -> None:
        """Transition the model to production."""
        self.client.set_registered_model_alias(
            name=model.name,
            alias=self.PRODUCTION_ALIAS,
            version=model.version,
        )

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
