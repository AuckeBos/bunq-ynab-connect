import os
import tempfile
from abc import abstractmethod
from logging import LoggerAdapter
from typing import List

import numpy as np
from kink import inject
from sklearn.base import ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.pipeline import Pipeline

import mlflow
from bunq_ynab_connect.classification.budget_category_encoder import (
    BudgetCategoryEncoder,
)
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
    """

    budget_id: str
    storage: AbstractStorage
    logger: LoggerAdapter
    run_id: str
    ids: List[str]
    label_encoder: BudgetCategoryEncoder

    @inject
    def __init__(self, budget_id: str, storage: AbstractStorage, logger: LoggerAdapter):
        self.storage = storage
        self.logger = logger
        self.budget_id = budget_id
        self.label_encoder = BudgetCategoryEncoder()

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

        Returns:
            X: List of BunqPayments
            y: List of YnabTransactions
        """
        transactions = self.storage.find(
            "matched_transactions",
            [("ynab_transaction.budget_id", "eq", self.budget_id)],
        )
        transactions = self.storage.rows_to_entities(transactions, MatchedTransaction)
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

    def create_pipeline(self, classifier: ClassifierMixin) -> Pipeline:
        """
        Create a pipeline with the given classifier
        """
        feature_extractor = FeatureExtractor()
        pipeline = Pipeline(
            [
                ("feature_extractor", feature_extractor),
                ("classifier", classifier),
            ]
        )
        return pipeline

    @abstractmethod
    def _run(self, transactions: List[MatchedTransaction]):
        """
        Run the actual experiment on the full set
        """
        raise NotImplementedError()
