from abc import abstractmethod
from logging import LoggerAdapter
from typing import List

import numpy as np
from kink import inject
from sklearn.calibration import LabelEncoder

import mlflow
from bunq_ynab_connect.classification.feature_extractor import FeatureExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class BasePaymentClassificationExperiment:
    """
    Base class for payment classification experiments.

    Each experiment should inherit from this class and implement the _run method.

    Attributes:
        budget_id: ID of the budget on which we are training a classifier
        storage: Storage to use for loading and saving data.
        logger: Logger to use for logging.
        run_id: ID of the current run. Set upon run()
        label_encoder: LabelEncoder to use for encoding the category names.
            Upon data load, the label encoder is fit on the complete y. This is to ensure
            that we know all labels at scoring time
    """

    budget_id: str
    storage: AbstractStorage
    logger: LoggerAdapter
    run_id: str
    label_encoder: LabelEncoder

    @inject
    def __init__(self, storage: AbstractStorage, logger: LoggerAdapter, budget_id: str):
        self.storage = storage
        self.logger = logger
        self.budget_id = budget_id
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """
        Load the dataset:
        - Load all matched transactions
        - Filter those for the current budget
        - Fit the label encoder on the complete y

        Returns:
            X: List of BunqPayments
            y: List of YnabTransactions
        """

        dataset = self.storage.find("matched_transactions")
        X = np.array([BunqPayment(**row["bunq_payment"]) for row in dataset])
        y = np.array([YnabTransaction(**row["ynab_transaction"]) for row in dataset])
        ids = [i for i in range(len(y)) if y[i].id == self.budget_id]
        # Todo: Set always empty
        X, y = X[ids], y[ids]
        self.label_encoder.fit([transaction.category_name for transaction in y])
        return X, y

    def run(self):
        """
        Run the experiment:
        - Load data
        - Enable autolog
        - Start run and _run
        """
        X, y = self.load_data()
        experiment_name = f"{self.__class__.__name__} [{self.budget_id}]"
        if not len(y):
            self.logger.info(
                f"Skipping experiment {experiment_name}, because no dataset was found"
            )
            return
        mlflow.sklearn.autolog()
        self.logger.info(f"Running experiment {experiment_name}")
        self.logger.info(f"Dataset has size {len(X)}")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            mlflow.set_tag("budget", self.budget_id)
            self.run_id = run.info.run_id
            return self._run(X, y)

    @abstractmethod
    def _run(self, X: List[BunqPayment], y: List[YnabTransaction]):
        """
        Run the actual experiment on the full set
        """
        raise NotImplementedError()
