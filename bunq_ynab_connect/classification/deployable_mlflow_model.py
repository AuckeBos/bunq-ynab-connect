import pickle

import pandas as pd
from kink import di
from mlflow.pyfunc import PythonModel
from sklearn.base import ClassifierMixin
from sklearn.calibration import LabelEncoder

import mlflow
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.bunq_payment import BunqPayment


class DeployableMlflowModel(PythonModel):
    """
    A PyFunc model that we can deploy.
    It loads the model and label encoder from the mlflow artifacts.
    During prediction, it stores the predictions in the database.
    """

    model: ClassifierMixin
    label_encoder: LabelEncoder

    def load_context(self, context):
        """
        Load the classifier and label encoder from the mlflow artifacts.
        """
        model_path = context.artifacts["model_path"]
        self.model = mlflow.sklearn.load_model(model_path)
        with open(context.artifacts["label_encoder_path"], "rb") as file:
            self.label_encoder = pickle.load(file)

    def predict(self, context, model_input, params=None):
        """
        Predict, and log the predictions in the database.
        """
        data = model_input.to_dict("records")
        predictions = self.model.predict(data)
        predictions = predictions.tolist()
        self.log_predictions(data, predictions)
        return self.label_encoder.inverse_transform(predictions)

    # Todo: Do this async
    def log_predictions(self, payments: list, predictions: list):
        """
        Log the predictions in the database.
        """
        data = [
            {
                "payment_id": payment["id"],
                "input": payment,
                "prediction": prediction,
            }
            for payment, prediction in zip(payments, predictions)
        ]
        storage = di[AbstractStorage]
        storage.upsert("payment_classifications", data)
