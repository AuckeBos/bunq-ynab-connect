from abc import abstractmethod
from datetime import datetime
from logging import LoggerAdapter

from kink import inject
from pendulum import now

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class AbstractDataset:
    """Abstract class for datasets.

    A Dataset can be updated, which means the data is loaded from storage, transformed
    into the format of the dataset, and then stored (upserted) in the storage.

    Attributes
    ----------
        NAME: The name of the dataset.
        KEY_COLUMN: The name of the column that is used as a key.
        last_runmoment: The last time the dataset was updated.

    """

    NAME: str
    KEY_COLUMN: str
    last_runmoment: datetime

    @inject
    def __init__(self, storage: AbstractStorage, logger: LoggerAdapter) -> None:
        self.storage = storage
        self.logger = logger

    @abstractmethod
    def load_new_data(self) -> list:
        """Based on self.last_runmoment, load the new data from the storage."""
        raise NotImplementedError

    def update(self) -> None:
        """Update the dataset.

        - Load the new data
        - Upsert the new data
        - Set the last runmoment
        """
        new_last_runmoment = now()
        self.last_runmoment = self.storage.get_last_runmoment(self.NAME)
        new_data = self.load_new_data()
        if len(new_data) > 0:
            self.logger.info("Upserting %s rows in %s", len(new_data), self.NAME)
            self.storage.upsert(self.NAME, new_data)
        self.storage.set_last_runmoment(self.NAME, new_last_runmoment)
        self.loggger.info("Upded dataset %s", self.NAME)
