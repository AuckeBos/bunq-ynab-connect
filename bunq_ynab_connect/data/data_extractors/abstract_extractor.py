from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime
from logging import LoggerAdapter

import pandas as pd
from kink import inject
from sqlalchemy import Engine
from sqlmodel import Session

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.general import now
from bunq_ynab_connect.models.bunq_account import BunqAccount


class AbstractExtractor(ABC):
    """Abstract class for data extractors.

    Attributes
    ----------
        destination: The name of the destination, used for logging and storage
        storage: The storage to save the data to
        logger: The logger to log to
        last_runmoment: The last runmoment of the extractor
        runmoment: The current runmoment of the extractor
        IS_FULL_LOAD: Whether the extractor is a full load extractor

    """

    destination: str
    storage: AbstractStorage
    logger: LoggerAdapter
    database: Engine
    last_runmoment: datetime
    runmoment: datetime

    IS_FULL_LOAD = False

    @inject
    def __init__(
        self,
        destination: str,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        database: Engine,
    ) -> None:
        self.destination = destination
        self.storage = storage
        self.logger = logger
        self.database = database

    @abstractmethod
    def load(self) -> Iterable:
        """Load the data from the source."""
        raise NotImplementedError

    def extract(self) -> None:
        """Extract the data from the bunq API, and save it in self.data.

        - Set current runmoment and load last runmoment
        - Load data from bunq API
        - Save data to storage. Upsert or ovewrite depending on IS_FULL_LOAD
        - Set last runmoment
        """
        self.runmoment = now()
        self.logger.info("Extracting %s", self.destination)
        self.last_runmoment = self.storage.get_last_runmoment(self.destination)
        data = self.load()
        if self.IS_FULL_LOAD:
            with Session(self.database) as session:
                new_data = [BunqAccount.model_validate(a) for a in data]
                session.add_all(new_data)
                session.commit()
            data_pd = pd.DataFrame.from_records(data)
            self.storage.overwrite(self.destination, data_pd)
        else:
            self.storage.upsert(self.destination, data)
        self.logger.info("Extracted %s items from %s", len(data), self.destination)
        self.storage.set_last_runmoment(self.destination, self.runmoment)
