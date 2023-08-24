from abc import ABC
from logging import LoggerAdapter
from typing import Iterable

from kink import inject

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class AbstractExtractor(ABC):
    destination: str
    storage: AbstractStorage
    logger: LoggerAdapter

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
        """
        data = self.load()
        self.storage.upsert(data, self.destination)
