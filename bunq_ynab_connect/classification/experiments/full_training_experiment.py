from logging import LoggerAdapter
from pathlib import Path
from typing import Any

import numpy as np
from kink import inject
from sklearn.base import ClassifierMixin
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import (
    StratifiedKFold,
)

import mlflow
from bunq_ynab_connect.classification.deployable_mlflow_model import (
    DeployableMlflowModel,
)
from bunq_ynab_connect.classification.experiments.base_payment_classification_experiment import (  # noqa: E501
    BasePaymentClassificationExperiment,
)
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.general import object_to_mlflow


class FullTrainingExperiment(BasePaymentClassificationExperiment):
    """An experiment to run a classifier with the best parameters on all data.

    The result of this experiment is the final model, which can be deployed.
    """

    model: ClassifierMixin
    RANDOM_STATE = 42

    @inject
    def __init__(
        self,
        budget_id: str,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        *,
        clf: Any,  # noqa: ANN401
        parameters: dict,
    ):
        super().__init__(budget_id, storage, logger)
        self.model = clf(**parameters)

    def _run(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        """Split the data once.

        Train the classifier on the training data and evaluate on the test data.
        Log the model as a DeployableMlflowModel.
        """
        classifier = self.create_pipeline(self.model)

        train_idx, test_idx = next(
            StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.RANDOM_STATE
            ).split(X, y)
        )
        X_train, X_test = X[train_idx], X[test_idx]  # noqa: N806
        y_train, y_test = y[train_idx], y[test_idx]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = cohen_kappa_score(y_test, y_pred)
        mlflow.log_metric("cohen_kappa", score)
        mlflow.sklearn.log_model(classifier, "model")
        object_to_mlflow(self.label_encoder, "label_encoder")

        artifact_uri = Path(mlflow.active_run().info.artifact_uri)
        artifacts = {
            "model_path": str(artifact_uri / "model"),
            "label_encoder_path": str(artifact_uri / "label_encoder"),
        }
        input_example = [X_train[0]]
        mlflow.pyfunc.log_model(
            artifact_path="deployable_model",
            python_model=DeployableMlflowModel(),
            artifacts=artifacts,
            input_example=input_example,
            registered_model_name=self.budget_id,
        )

        self.logger.info("Finished training")
