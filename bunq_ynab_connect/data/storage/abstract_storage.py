from abc import ABC, abstractmethod
from datetime import datetime
from logging import LoggerAdapter
from typing import Any, List, Optional, Tuple

import pandas as pd
from kink import inject
from prefect.logging.loggers import PrefectLogAdapter

from bunq_ynab_connect.data.metadata import Metadata
from bunq_ynab_connect.helpers.general import now


class AbstractStorage(ABC):
    """
    AbstractStorage is an abstract class that defines the interface for a storage class.
    This class itself also defines some common methods that can be used by all storage classes.

    Attributes:
        metadata: The metadata class. Used to get information about the tables.
        logger: The logger to use to log messages.
    """

    metadata: Metadata
    logger: LoggerAdapter

    def __init__(self, metadata: Metadata, logger: LoggerAdapter) -> None:
        self.metadata = metadata
        self.logger = logger

    @abstractmethod
    def _upsert(self, data: List, table: str, key_col: str, timestamp_col: str) -> None:
        """
        Upsert a list of rows into a table. Use the key_col to identify the row and the timestamp_col to
        determine the order of the rows.
        """
        raise NotImplementedError

    @abstractmethod
    def _insert(self, data: List, table: str) -> None:
        """
        Insert a list of rows into a table.
        """
        raise NotImplementedError

    @abstractmethod
    def _overwrite(self, data: pd.DataFrame, table: str) -> None:
        """
        Overwrite the full contents of a table with the data.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_query(self, query: List[Tuple] = None) -> Any:
        """
        Convert a query to the appropriate format for the storage class.
        """
        raise NotImplementedError

    @abstractmethod
    def find(
        self,
        table: str,
        query: List[Tuple] = None,
        sort: List[str] = None,
        asc: bool = True,
    ) -> List:
        """
        Find rows in a table that match the query.

        Parameters:
            table: The name of the table to query.
            query: A list of queries. Each query is a tuple of (column, operator, value). The operator is one of the
                following: eq, gt, gte, in, lt, lte, ne, nin. Implementations should convert this to the
                appropriate query.
            sort: A list of columns to sort by.
            asc: Whether to sort ascending or descending.
        """
        raise NotImplementedError

    def get(self, table: str, as_dataframe: bool = False) -> List:
        """
        Get all rows in a table.

        Parameters:
            table: The name of the table to query.
        """
        result = self.find(table)
        if as_dataframe:
            result = pd.DataFrame(result)
        return result

    def overwrite(self, data: pd.DataFrame, table: str) -> None:
        """
        Overwrite the full contents of a table with a dataframe.
        """
        self._overwrite(data, table)

    def upsert(self, data: List, table_name: str) -> None:
        """
        Add updated_at, and then call _upsert.
        """
        updated_at = now().isoformat()
        data = map(lambda x: {**x, "updated_at": updated_at}, data)
        table = self.metadata.get_table(table_name)
        self._upsert(data, table_name, table.key_col, table.timestamp_col)

    def insert(self, data: List, table: str) -> None:
        """
        Add inserted_at, and then call _insert.
        """
        inserted_at = now().isoformat()
        data = map(lambda x: {**x, "inserted_at": inserted_at}, data)
        self._insert(data, table)

    def find_one(
        self, table: str, query: dict, sort: List[str] = [], asc: bool = True
    ) -> Optional[dict]:
        """
        Find one row in a table that matches the query.
        """
        result = self.find(table, query, sort, asc)
        return result[0] if result else None

    def get_last_runmoment(self, source: str) -> datetime:
        """
        Get the last timestamp from the runmoments table. This is used to determine the window of data
        to load. Return 2020-01-01 if there is no timestamp in the runmoments table.
        """
        result = self.find_one("runmoments", [("source", "eq", source)])
        return (
            datetime.fromisoformat(result["timestamp"])
            if result
            else datetime(2020, 1, 1)
        )

    def set_last_runmoment(self, source: str, timestamp: datetime) -> None:
        """
        Set the last timestamp in the runmoments table. Use the upsert method.
        """
        data = [{"source": source, "timestamp": timestamp.isoformat()}]
        self.upsert(data, "runmoments")
        self.logger.info(f"Updated runmoment of {source} to {timestamp}")

    def get_window(self, source: str) -> Tuple[datetime, datetime]:
        """
        Get the window of data to load. This is the last timestamp in the runmoments table and the
        current time.
        """
        start = self.get_last_runmoment(source)
        end = datetime.now()
        self.logger.info(f"Retrieved window for {source} as {start} - {end}")
        return start, end
