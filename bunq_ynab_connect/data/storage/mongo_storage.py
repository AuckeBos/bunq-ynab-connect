from ast import Tuple
from logging import LoggerAdapter
from typing import Any, List

import pandas as pd
from kink import inject
from prefect.logging.loggers import PrefectLogAdapter
from pymongo import MongoClient
from pymongo.database import Database

from bunq_ynab_connect.data.metadata import Metadata
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class MongoStorage(AbstractStorage):
    """
    The MongoStorage class is used to store data in a MongoDB database.

    Attributes:
        client: The MongoDB client.
        database: The MongoDB database.
    """

    client: MongoClient
    database: Database

    @inject
    def __init__(
        self,
        client: MongoClient,
        database: Database,
        metadata: Metadata,
        logger: LoggerAdapter,
    ):
        """
        Create a connection to the MongoDB database.
        """
        super().__init__(metadata, logger)
        self.client = client
        self.database = database
        self.test_connection()
        self.set_indexes()

    def convert_query(self, query: List[Tuple] = None) -> Any:
        """
        Each operator is simply prefixed with a $. The list is converted to a dictionary.
        """
        query = query or []
        return {q[0]: {f"${q[1]}": q[2]} for q in query}

    def find(
        self,
        table: str,
        query: List[Tuple] = None,
        sort: List[str] = None,
        asc: bool = True,
    ) -> List:
        """
        Find rows in a table that match the query. A query is a dictionary of key-equals-value pairs.
        """
        sort = sort or []
        query = self.convert_query(query)
        sort = [(key, 1 if asc else -1) for key in sort]
        result = self.database[table].find(query)
        if sort:
            result = result.sort(sort)
        return list(result)

    def _upsert(self, table: str, data: List, key_col: str, timestamp_col: str) -> None:
        """
        Upsert each item. For now, simply loop over them and insert each one separately.
        Also add an updated_at column.

        Todo: Speed up
        """
        table = self.database[table]
        for row in data:
            table.update_one({key_col: row[key_col]}, {"$set": row}, upsert=True)

    def _insert(self, table: str, data: List) -> None:
        """
        Insert all items in the data list.
        Also add an inserted_at column.
        """
        if data:
            table = self.database[table]
            table.insert_many(data)

    def _overwrite(self, table: str, data: pd.DataFrame) -> None:
        """
        Overwrite the full contents of a table with the data.
        """
        with self.client.start_session() as session:
            with session.start_transaction():
                self.database[table].drop()
                self._insert(table, data.to_dict("records"))

    def delete(self, table: str, query: List[Tuple] = None) -> None:
        """
        Delete rows in a table that match the query.

        Parameters:
            table: The name of the table to query.
            query: A list of queries. Each query is a tuple of (column, operator, value). The operator is one of the
                following: eq, gt, gte, in, lt, lte, ne, nin. Implementations should convert this to the
                appropriate query.
        """
        query = self.convert_query(query)
        self.database[table].delete_many(query)

    def count(self, table: str, query: List[Tuple] = None) -> int:
        """
        Count the number of rows in a table that match the query.

        Parameters:
            table: The name of the table to query.
            query: A list of queries. Each query is a tuple of (column, operator, value). The operator is one of the
                following: eq, gt, gte, in, lt, lte, ne, nin. Implementations should convert this to the
                appropriate query.
        """
        query = self.convert_query(query)
        return self.database[table].count_documents(query)

    def test_connection(self) -> None:
        """
        Test if the client is connected to the database.
        """
        try:
            self.client.server_info()
        except Exception as e:
            self.logger.error(e)
            raise Exception("Could not connect to MongoDB. Is the server running?")

    def set_indexes(self) -> None:
        """
        Set indexes on the tables.
        """
