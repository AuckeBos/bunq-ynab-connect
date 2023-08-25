from abc import ABC
from datetime import datetime
from logging import LoggerAdapter
from typing import Iterable

from kink import inject

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class AbstractExtractor(ABC):
    """
    Abstract class for data extractors.

    Attributes:
        destination: The name of the destination, used for logging and storage
        storage: The storage to save the data to
        logger: The logger to log to
        last_runmoment: The last runmoment of the extractor
        runmoment: The current runmoment of the extractor
    """

    destination: str
    storage: AbstractStorage
    logger: LoggerAdapter
    last_runmoment: datetime
    runmoment: datetime

    @inject
    def __init__(
        self, destination: str, storage: AbstractStorage, logger: LoggerAdapter
    ) -> None:
        self.destination = destination
        self.storage = storage
        self.logger = logger

    def load(self) -> Iterable:
        """
        Load the data from the source.
        """
        raise NotImplementedError

    def extract(self) -> None:
        """
        Extract the data from the bunq API, and save it in self.data.
        - Set current runmoment and load last runmoment
        - Load data from bunq API
        - Save data to storage
        - Set last runmoment
        """
        self.runmoment = datetime.now()
        self.logger.info(f"Extracting {self.destination}")
        self.last_runmoment = self.storage.get_last_runmoment(self.destination)
        data = self.load()
        self.storage.upsert(data, self.destination)
        self.logger.info(f"Extracted {len(data)} items from {self.destination}")
        self.storage.set_last_runmoment(self.destination, self.runmoment)
