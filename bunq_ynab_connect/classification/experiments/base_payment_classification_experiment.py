from abc import abstractmethod
from logging import LoggerAdapter
from typing import List

from kink import inject

import mlflow
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class BasePaymentClassificationExperiment:
    """ """

    storage: AbstractStorage
    logger: LoggerAdapter
    run_id: str

    @inject
    def __init__(self, storage: AbstractStorage, logger: LoggerAdapter):
        self.storage = storage
        self.logger = logger

    def load_data(self):
        dataset = self.storage.find("matched_transactions")
        X = [BunqPayment(**row["bunq_payment"]) for row in dataset]
        y = [YnabTransaction(**row["ynab_transaction"]) for row in dataset]
        self.logger.info(f"Loaded dataset with {len(X)} rows")
        return X, y

    def run(self):
        X, y = self.load_data()
        mlflow.sklearn.autolog()
        mlflow.set_experiment(self.__class__.__name__)
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            return self._run(X, y)

    @abstractmethod
    def _run(self, X: List[BunqPayment], y: List[YnabTransaction]):
        raise NotImplementedError()
