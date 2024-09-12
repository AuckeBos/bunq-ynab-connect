import pickle
from logging import LoggerAdapter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from kink import di
from sklearn.base import ClassifierMixin

import mlflow
from bunq_ynab_connect.classification.budget_category_encoder import (
    BudgetCategoryEncoder,
)
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from mlflow.models import set_model
from mlflow.pyfunc import PythonModel, PythonModelContext


class DeployableMlflowModel(PythonModel):
    """A PyFunc model that we can deploy.

    It loads the model and label encoder from the mlflow artifacts.
    During prediction, it stores the predictions in the database.
    """

    model: ClassifierMixin | None = None
    label_encoder: BudgetCategoryEncoder | None = None

    def load_context(self, context: PythonModelContext) -> None:
        """Load the classifier and label encoder from the mlflow artifacts."""
        self.model = mlflow.sklearn.load_model(context.artifacts["model_path"])
        with Path.open(context.artifacts["label_encoder_path"], "rb") as file:
            self.label_encoder = pickle.load(file)

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame | list[dict],
        __: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Predict, and log the predictions in the database."""
        if isinstance(model_input, pd.DataFrame):
            data = model_input.to_dict("records")
        else:
            data = model_input
        if not self.model:
            logger = di[LoggerAdapter]
            logger.warning(
                "Model context not loaded. Cannot make predictions. Context: %s",
                context,
            )
            return [None]
        predictions = [
            {
                "category_name": self.label_encoder.id_to_name[prediction],
                "category_id": prediction,
            }
            for prediction in self.label_encoder.inverse_transform(
                self.model.predict(data).tolist()
            )
        ]
        self.log_predictions(data, predictions)
        return np.array(predictions)

    # TODO: Do this async
    # TODO: Add model version identifier
    def log_predictions(self, payments: list, predictions: list) -> None:
        data = [
            {
                "payment_id": payment["id"],
                "input": payment,
                "prediction": prediction,
            }
            for payment, prediction in zip(payments, predictions, strict=True)
        ]
        storage = di[AbstractStorage]
        storage.insert("payment_classifications", data)


set_model(DeployableMlflowModel())
