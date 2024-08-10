from __future__ import annotations

import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from kink import inject
from sklearn.pipeline import Pipeline

import mlflow
from bunq_ynab_connect.classification.budget_category_encoder import (
    BudgetCategoryEncoder,
)
from bunq_ynab_connect.classification.feature_extractor import FeatureExtractor
from bunq_ynab_connect.models.matched_transaction import MatchedTransaction

if TYPE_CHECKING:
    from logging import LoggerAdapter

    from sklearn.base import ClassifierMixin

    from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class BasePaymentClassificationExperiment:
    """Base class for payment classification experiments.

    Each experiment should inherit from this class and implement the _run method.

    Attributes
    ----------
        budget_id: ID of the budget on which we are training a classifier
        storage: Storage to use for loading and saving data.
        logger: Logger to use for logging.
        parent_run_id: ID of the current run. Set upon run()
        ids: List of IDs of the matched transactions used for training
        label_encoder: LabelEncoder used to encode the categories

    """

    budget_id: str
    storage: AbstractStorage
    logger: LoggerAdapter
    parent_run_id: str
    ids: list[str]
    label_encoder: BudgetCategoryEncoder

    @inject
    def __init__(self, budget_id: str, storage: AbstractStorage, logger: LoggerAdapter):
        self.storage = storage
        self.logger = logger
        self.budget_id = budget_id
        self.label_encoder = BudgetCategoryEncoder()

    def log_transactions(
        self, transactions: list[MatchedTransaction], name: str
    ) -> None:
        """Log the training data to mlflow."""
        with tempfile.TemporaryDirectory() as dir_:
            filename = f"{name}.txt"
            path = Path(dir_) / filename
            with path.open() as file:
                ids = [t.match_id for t in transactions]
                file.write("\n".join(ids))
            mlflow.log_artifact(path)
            mlflow.log_text(str(len(ids)), f"len_{name}.txt")

    def load_data(self) -> list[MatchedTransaction]:
        """Load the dataset.

        - Load all matched transactions for the given budget
        - Convert them to MatchedTransaction entities

        Returns
        -------
            List of MatchedTransaction entities

        """
        transactions = self.storage.find(
            "matched_transactions",
            [("ynab_transaction.budget_id", "eq", self.budget_id)],
        )
        return self.storage.rows_to_entities(transactions, MatchedTransaction)

    def transactions_to_xy(
        self, transactions: list[MatchedTransaction]
    ) -> tuple[np.array, np.array]:
        """Convert a list of MatchedTransactions to X and y.

        Returns
        -------
            X: Array of bunq payments
            y: Array of categories as integers

        """
        X = np.array([t.bunq_payment.dict() for t in transactions])  # noqa: N806
        y = np.array([t.ynab_transaction.dict() for t in transactions])
        y = self.label_encoder.fit_transform(y)
        return X, y

    def run(self) -> None:
        """Run the experiment.

        - Load data
        - Enable autolog
            Skip logging of models, because this takes a lot of space
        - Start run and _run
        """
        transactions = self.load_data()
        experiment_name = self.get_experiment_name()
        if not len(transactions):
            self.logger.info(
                "Skipping experiment %s, because no dataset was found", experiment_name
            )
            return
        X, y = self.transactions_to_xy(transactions)  # noqa: N806
        self.logger.info("Running experiment %s", experiment_name)
        self.logger.info("Dataset has size %s", len(transactions))
        mlflow.set_experiment(experiment_name)
        mlflow.sklearn.autolog(log_models=False)
        with mlflow.start_run() as run:
            self.log_transactions(transactions, "full_set_ids")

            mlflow.set_tag("budget", self.budget_id)
            self.parent_run_id = run.info.run_id
            self._run(X, y)

    def create_pipeline(self, classifier: ClassifierMixin) -> Pipeline:
        feature_extractor = FeatureExtractor()
        return Pipeline(
            [
                ("feature_extractor", feature_extractor),
                ("classifier", classifier),
            ]
        )

    def get_experiment_name(self) -> str:
        return f"{self.__class__.__name__} [{self.budget_id}]"

    @abstractmethod
    def _run(self, X: np.array, y: np.array) -> None:  # noqa: N803
        """Run the actual experiment on the full set."""
        raise NotImplementedError
