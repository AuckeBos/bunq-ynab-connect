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

    def _upsert(self, data: List, table: str, key_col: str, timestamp_col: str) -> None:
        """
        Upsert each item. For now, simply loop over them and insert each one separately.
        Also add an updated_at column.

        Todo: Speed up
        """
        table = self.database[table]
        for row in data:
            table.update_one({key_col: row[key_col]}, {"$set": row}, upsert=True)

    def _insert(self, data: List, table: str) -> None:
        """
        Insert all items in the data list.
        Also add an inserted_at column.
        """
        if data:
            table = self.database[table]
            table.insert_many(data)

    def _overwrite(self, data: pd.DataFrame, table: str) -> None:
        """
        Overwrite the full contents of a table with the data.
        """
        with self.client.start_session() as session:
            with session.start_transaction():
                self.database[table].drop()
                self._insert(data.to_dict("records"), table)

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
