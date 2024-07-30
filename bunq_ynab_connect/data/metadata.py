import json

from kink import inject

from bunq_ynab_connect.data.table_metadata import TableMetadata
from bunq_ynab_connect.helpers.config import METADATA_DIR


@inject
class Metadata:
    """Metadata is a class that contains metadata about the tables in the database.

    It loads the metadata from JSON files in the metadata directory.
    It is used by the storage classes to get information about the tables, based on the
        table name.
    """

    def __init__(self) -> None:
        """Load the metadata from the JSON files in the metadata directory."""
        self.tables = [
            TableMetadata(**json.load(f.open()))
            for f in (METADATA_DIR / "tables").glob("*.json")
        ]

    def get_table(self, table: str) -> TableMetadata:
        """Get the metadata for a table."""
        try:
            return next(x for x in self.tables if x.name == table)
        except StopIteration as e:
            msg = f"Table {table} not found in metadata."
            raise Exception(msg) from e  # noqa: TRY002
