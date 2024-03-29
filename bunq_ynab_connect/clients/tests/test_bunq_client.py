import datetime
import json
from logging import getLogger
from unittest import mock
from unittest.mock import Mock

import pytest
from bunq.sdk.model.generated.endpoint import MonetaryAccountBank, Payment
from pytest import fixture

from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage

ACCOUNT_ID_FOR_TESTING = 4023038


@fixture
def client() -> BunqClient:
    return BunqClient(storage=Mock(spec=AbstractStorage), logger=getLogger("test"))


@pytest.mark.skip(reason="Not working in CI/CD")
def test_get_accounts(client: BunqClient):
    # Arrange
    # Act
    accounts = client.get_accounts()
    # Assert
    assert len(accounts) > 0


@pytest.mark.skip(reason="Not working in CI/CD")
def test_get_payments(client: BunqClient):
    # Arrange
    # Act
    account = MonetaryAccountBank.from_json(json.dumps({}))
    with mock.patch.object(MonetaryAccountBank, "_id_", ACCOUNT_ID_FOR_TESTING):
        # do stuff
        payments = client.get_payments_for_account(
            account,
            last_runmoment=datetime.datetime.today() - datetime.timedelta(weeks=1),
        )
    # Assert
    assert len(payments) > 0
    assert type(payments[0]) == Payment
