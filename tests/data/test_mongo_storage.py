from datetime import datetime, timezone
from logging import LoggerAdapter
from time import sleep
from unittest.mock import Mock

import mongomock
import pytest
from kink import di

from bunq_ynab_connect.data.metadata import Metadata
from bunq_ynab_connect.data.storage.mongo_storage import MongoStorage
from bunq_ynab_connect.data.table_metadata import TableMetadata


@pytest.fixture()
def mongo() -> mongomock.MongoClient:
    """Return a mongomock client."""
    return mongomock.MongoClient()


@pytest.fixture()
def metadata() -> Metadata:
    """Return a mock metadata object."""
    return Mock(spec=Metadata)


@pytest.fixture()
def mongo_storage(monkeypatch, mongo, metadata) -> MongoStorage:  # noqa: ANN001
    """Return a MongoStorage object."""
    monkeypatch.setattr(MongoStorage, "set_indexes", Mock())
    storage = MongoStorage(mongo, mongo["test_database"], metadata, di[LoggerAdapter])
    storage.metadata.get_table.return_value = TableMetadata(
        name="test_table", key_col="key", timestamp_col="timestamp", type="test_table"
    )
    return storage


def test_insert_and_find_one(mongo_storage: MongoStorage) -> None:
    """Test that inserting data into a table and then finding it works.

    Also test that the inserted_at column is added.
    """
    # Arrange
    current_time = datetime.now(tz=timezone.utc)
    table_name = "test_table"
    data = [{"key": 1, "value": "one"}, {"key": 2, "value": "two"}]

    # Act
    mongo_storage.insert(table_name, data)
    result = mongo_storage.find_one(table_name, [("key", "eq", 2)])

    # Assert
    assert result["key"] == 2  # noqa: PLR2004
    assert datetime.fromisoformat(result["inserted_at"]) > current_time


def test_upsert_and_find(mongo_storage: MongoStorage) -> None:
    """Test that upserting data into a table and then finding it works."""
    # Arrange
    table_name = "test_table"
    data = [
        {"key": 1, "value": "one"},
        {"key": 2, "value": "two"},
    ]

    # Act
    mongo_storage.upsert(table_name, data)
    result = mongo_storage.find(table_name, [("key", "in", [1, 2])], ["key"], asc=False)

    # Assert
    assert len(result) == 2  # noqa: PLR2004
    assert result[0]["key"] == 2  # noqa: PLR2004
    assert result[1]["key"] == 1


def test_upsert_does_update(mongo_storage: MongoStorage) -> None:
    """Test that upserting data into a table updates the data."""
    # Arrange
    table_name = "test_table"
    data = [{"key": 1, "value": "one"}, {"key": 3, "value": "two"}]

    # Act
    mongo_storage.insert(table_name, data)
    data = [
        {"key": 1, "value": "three"},
    ]
    mongo_storage.upsert(table_name, data)
    result = mongo_storage.find(table_name, [("key", "ne", 3)], ["key"], asc=False)

    # Assert
    assert len(result) == 1
    assert result[0]["key"] == 1
    assert result[0]["value"] == "three"


def test_get_window(mongo_storage: MongoStorage) -> None:
    """Test that the window is set and retrieved correctly."""
    # Arrange
    source = "test_source"
    current_time = datetime.now(tz=timezone.utc).replace(microsecond=0)

    mongo_storage.metadata.get_table().key_col = "source"

    # Act
    mongo_storage.set_last_runmoment(source, current_time)
    start, end = mongo_storage.get_window(source)

    # Assert
    assert start == current_time
    assert end > current_time


def test_get_last_runmoment(mongo_storage: MongoStorage) -> None:
    """Test that the last runmoment is set and retrieved correctly."""
    # Arrange
    source = "test_source"
    current_time = datetime.now(tz=timezone.utc).replace(microsecond=0)

    mongo_storage.metadata.get_table().key_col = "source"

    # Act
    mongo_storage.set_last_runmoment(source, current_time)
    result = mongo_storage.get_last_runmoment(source)

    # Assert
    assert result == current_time


def test_insert_and_get(mongo_storage: MongoStorage) -> None:
    """Test that inserting data into a table and then getting it works."""
    # Arrange
    table_name = "test_table"
    data = [
        {"key": 1, "value": "one"},
        {"key": 2, "value": "two"},
    ]

    # Act
    mongo_storage.insert(table_name, data)
    result = mongo_storage.get(table_name)

    # Assert
    assert len(result) == 2  # noqa: PLR2004


def test_inserted_at_is_added(mongo_storage: MongoStorage) -> None:
    """Test that the inserted_at column is added when inserting data."""
    # Arrange
    table_name = "test_table"
    data = [
        {"key": 1, "value": "one"},
    ]

    # Act
    mongo_storage.insert(table_name, data)
    result = mongo_storage.get(table_name)

    # Assert
    assert len(result) == 1
    assert "inserted_at" in result[0]


def test_updated_at_is_updated(mongo_storage: MongoStorage) -> None:
    """Test that the updated_at column is updated when upserting data."""
    # Arrange
    table_name = "test_table"
    data = [
        {"key": 1, "value": "one"},
    ]

    # Act
    mongo_storage.upsert(table_name, data)
    old_updated_at = mongo_storage.get(table_name)[0]["updated_at"]
    sleep(0.1)
    mongo_storage.upsert(table_name, data)
    new_updated_at = mongo_storage.get(table_name)[0]["updated_at"]

    # Assert
    assert new_updated_at > old_updated_at


def test_convert_query_is_used(mongo_storage: MongoStorage) -> None:
    """Test that the convert_query method is called when find is called."""
    # Arrange
    table_name = "test_table"
    query = [("key", "eq", 1)]
    mongo_storage.convert_query = Mock()
    mongo_storage.convert_query.return_value = {"key": {"$eq": 1}}

    # Act
    mongo_storage.find(table_name, query)

    # Assert
    mongo_storage.convert_query.assert_called_once_with(query)
