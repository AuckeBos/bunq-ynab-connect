import os
import tempfile
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
from bunq_ynab_connect.models.ynab.matched_transaction import MatchedTransaction
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
    ids: List[str]

    @inject
    def __init__(self, storage: AbstractStorage, logger: LoggerAdapter, budget_id: str):
        self.storage = storage
        self.logger = logger
        self.budget_id = budget_id
        self.label_encoder = LabelEncoder()

    def log_transactions(self, transactions: List[MatchedTransaction], name: str):
        """
        Log the training data to mlflow
        """
        with tempfile.TemporaryDirectory() as dir:
            filename = f"{name}.txt"
            path = os.path.join(dir, filename)
            with open(path, "w") as file:
                ids = [t.match_id for t in transactions]
                file.write("\n".join(ids))
            mlflow.log_artifact(path)
            mlflow.log_text(str(len(ids)), f"len_{name}.txt")

    def load_data(self) -> List[MatchedTransaction]:
        """
        Load the dataset:
        - Load all matched transactions
        - Filter those for the current budget
        - Fit the label encoder on the complete y

        Returns:
            X: List of BunqPayments
            y: List of YnabTransactions
        """
        transactions = self.storage.find(
            "matched_transactions",
            [("ynab_transaction.budget_id", "eq", self.budget_id)],
        )
        transactions = self.storage.rows_to_entities(transactions, MatchedTransaction)
        self.label_encoder.fit([t.ynab_transaction.category_name for t in transactions])
        return transactions

    def run(self):
        """
        Run the experiment:
        - Load data
        - Enable autolog
        - Start run and _run
        """
        transactions = self.load_data()
        experiment_name = f"{self.__class__.__name__} [{self.budget_id}]"
        if not len(transactions):
            self.logger.info(
                f"Skipping experiment {experiment_name}, because no dataset was found"
            )
            return
        mlflow.sklearn.autolog()
        self.logger.info(f"Running experiment {experiment_name}")
        self.logger.info(f"Dataset has size {len(transactions)}")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            self.log_transactions(transactions, "full_set_ids")
            mlflow.set_tag("budget", self.budget_id)
            self.run_id = run.info.run_id
            self._run(transactions)

    @abstractmethod
    def _run(self, transactions: List[MatchedTransaction]):
        """
        Run the actual experiment on the full set
        """
        raise NotImplementedError()
