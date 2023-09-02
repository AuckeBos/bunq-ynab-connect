import json
from abc import ABC, abstractmethod
from datetime import datetime
from logging import LoggerAdapter
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from bunq.sdk.model.generated.endpoint import (
    BunqResponsePaymentList,
    MonetaryAccountBank,
    Payment,
)
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
        RUNMOMENT_START: The default start date for the runmoments table.
        METADATA_COLUMNS: The columns that are created in this class,
            and should be excluded when converting a dict to a model.
    """

    metadata: Metadata
    logger: LoggerAdapter
    RUNMOMENT_START = datetime(2020, 1, 1)
    METADATA_COLUMNS = ["_id", "updated_at"]

    def __init__(self, metadata: Metadata, logger: LoggerAdapter) -> None:
        self.metadata = metadata
        self.logger = logger

    @abstractmethod
    def _upsert(self, table: str, data: List, key_col: str, timestamp_col: str) -> None:
        """
        Upsert a list of rows into a table. Use the key_col to identify the row and the timestamp_col to
        determine the order of the rows.
        """
        raise NotImplementedError

    @abstractmethod
    def _insert(self, table: str, data: List) -> None:
        """
        Insert a list of rows into a table.
        """
        raise NotImplementedError

    @abstractmethod
    def _overwrite(self, table: str, data: pd.DataFrame) -> None:
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

    @abstractmethod
    def delete(self, table: str, query: List[Tuple] = None) -> None:
        """
        Delete rows in a table that match the query.

        Parameters:
            table: The name of the table to query.
            query: A list of queries. Each query is a tuple of (column, operator, value). The operator is one of the
                following: eq, gt, gte, in, lt, lte, ne, nin. Implementations should convert this to the
                appropriate query.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self, table: str, query: List[Tuple] = None) -> int:
        """
        Count the number of rows in a table that match the query.

        Parameters:
            table: The name of the table to query.
            query: A list of queries. Each query is a tuple of (column, operator, value). The operator is one of the
                following: eq, gt, gte, in, lt, lte, ne, nin. Implementations should convert this to the
                appropriate query.
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

    def overwrite(self, table: str, data: pd.DataFrame) -> None:
        """
        Overwrite the full contents of a table with a dataframe.
        """
        self._overwrite(table, data)

    def upsert(self, table_name: str, data: List) -> None:
        """
        Add updated_at, and then call _upsert.
        """
        updated_at = now().isoformat()
        data = map(lambda x: {**x, "updated_at": updated_at}, data)
        table = self.metadata.get_table(table_name)
        self._upsert(table_name, data, table.key_col, table.timestamp_col)

    def insert_if_not_exists(self, table_name: str, data: List) -> None:
        """
        Check if the data already exists in the table. If not, insert it.
        """
        table = self.metadata.get_table(table_name)
        data = [
            d
            for d in data
            if not self.find_one(table_name, [(table.key_col, "eq", d[table.key_col])])
        ]
        self.insert(table_name, data)

    def insert(self, table: str, data: List) -> None:
        """
        Add inserted_at, and then call _insert.

        Parameters:
            table: The name of the table to insert into.

        """
        if not data:
            self.logger.info(f"No data to insert into {table}")
            return
        inserted_at = now().isoformat()
        data = map(lambda x: {**x, "inserted_at": inserted_at}, data)
        self._insert(table, data)

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
        last_runmoment = self.find_one("runmoments", [("source", "eq", source)])
        result = (
            datetime.fromisoformat(last_runmoment["timestamp"])
            if last_runmoment
            else self.RUNMOMENT_START
        )
        self.logger.info(f"Retrieved last runmoment of {source} as {result}")
        return result

    def set_last_runmoment(self, source: str, timestamp: datetime) -> None:
        """
        Set the last timestamp in the runmoments table. Use the upsert method.
        """
        data = [{"source": source, "timestamp": timestamp.isoformat()}]
        self.upsert("runmoments", data)
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

    def rows_to_entities(self, rows: List[dict], fn: Callable, as_json: bool = False):
        """
        Convert a list of rows to a list of entities.

        Parameters:
            rows: The rows to convert.
            fn: The function to use to convert the data to an entity.
            as_json: Whether the dict should be provided as json string to the function.
                Else the dict is provided as kwargs.
        """
        rows = [
            {k: v for k, v in d.items() if not k in self.METADATA_COLUMNS} for d in rows
        ]
        if as_json:
            rows_as_entity = [fn(json.dumps(d)) for d in rows]
        else:
            rows_as_entity = [fn(**d) for d in rows]
        return rows_as_entity

    def get_as_entity(self, table: str, fn: Callable, as_json: bool = False):
        """
        Get the data from a table as an entity. Use rows_to_entities to convert the rows to entities.

        Parameters:
            table: The name of the table to query.
            fn: The function to use to convert the data to an entity.
            as_json: Whether the dict should be provided as json string to the function.
                Else the dict is provided as kwargs.
        """
        rows = self.get(table)
        return self.rows_to_entities(rows, fn, as_json)

    def get_budget_ids(self) -> List[str]:
        """
        Get a list of all budget ids
        """
        ynab_budgets = self.find("ynab_budgets")
        return [budget["id"] for budget in ynab_budgets]
