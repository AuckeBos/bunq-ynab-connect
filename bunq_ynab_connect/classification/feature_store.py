from logging import LoggerAdapter

from kink import inject

from bunq_ynab_connect.classification.datasets.matched_transactions_dataset import (
    MatchedTransactionsDataset,
)
from bunq_ynab_connect.data.metadata import Metadata
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


@inject
class FeatureStore:
    """A FeatureStore can load part of a dataset based on a list of IDs.

    The list of IDs should be stored along with a model, such that
    the trainingset can always be reproduced.
    """

    storage: AbstractStorage
    logger: LoggerAdapter
    metadata: Metadata

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, metadata: Metadata
    ):
        self.storage = storage
        self.logger = logger
        self.metadata = metadata

    def dataset_by_ids(self, dataset: str, ids: list) -> list:
        metadata = self.metadata.get_table(dataset)
        query = [(metadata.key_col, "in", ids)]
        return self.storage.find(metadata.name, query)

    def update(self) -> None:
        """Update the feature store.

        Load all datasets and upsert them.
        """
        self.logger.info("Updating feature store")
        datasets = [MatchedTransactionsDataset()]
        for dataset in datasets:
            dataset.update()
        self.logger.info("Updated feature store")
