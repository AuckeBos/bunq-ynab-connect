from datetime import datetime
from unittest.mock import Mock

import pytest
from bunq.sdk.model.generated import endpoint
from bunq.sdk.model.generated.endpoint import BunqResponsePaymentList

from bunq_ynab_connect.clients.bunq_client import (
    BunqClient,
    MonetaryAccountBank,
    Payment,
)


# Mocking the endpoint.Payment.list function
def mock_payment_list(*args, **kwargs):
    mock_payment_1 = Mock()
    mock_payment_1.created = "2023-08-29T12:00:00"

    mock_payment_2 = Mock()
    mock_payment_2.created = "2023-08-28T12:00:00"
    mock_payment_list = [mock_payment_1, mock_payment_2]
    pagination = Mock()
    pagination.has_previous_page.return_value = False

    payment_list = Mock()
    payment_list.value = mock_payment_list
    payment_list.pagination = pagination
    return payment_list


@pytest.fixture
def mock_bunq_response(monkeypatch):
    monkeypatch.setattr(endpoint.Payment, "list", mock_payment_list)


def test_get_payments_for_account_returns_all_payments(mock_bunq_response):
    # Prepare mock data
    mock_account = Mock(spec=MonetaryAccountBank)
    mock_last_runmoment = datetime(2023, 8, 27)

    # Create an instance of BunqClient with mock storage and logger
    bunq_client = BunqClient(Mock(), Mock())

    # Call the method under test
    payments = bunq_client.get_payments_for_account(
        account=mock_account, last_runmoment=mock_last_runmoment
    )

    # Assertions
    assert len(payments) == 2
    assert payments[0].created == "2023-08-29T12:00:00"
    assert payments[1].created == "2023-08-28T12:00:00"


def test_only_payments_after_last_runmoment_are_returned(mock_bunq_response):
    # Prepare mock data
    mock_account = Mock(spec=MonetaryAccountBank)
    mock_last_runmoment = datetime(2023, 8, 28, 13)

    # Create an instance of BunqClient with mock storage and logger
    bunq_client = BunqClient(Mock(), Mock())

    # Call the method under test
    payments = bunq_client.get_payments_for_account(
        account=mock_account, last_runmoment=mock_last_runmoment
    )

    # Assertions
    assert len(payments) == 1
    assert payments[0].created == "2023-08-29T12:00:00"
