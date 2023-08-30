from unittest.mock import Mock

import pytest
import ynab

from bunq_ynab_connect.clients.ynab_client import YnabClient


def test_get_account_for_budget_calls_api(monkeypatch):
    api = Mock(spec=ynab.AccountsApi)
    api.get_accounts.return_value = Mock(data=Mock(accounts=[]))
    monkeypatch.setattr(ynab, "AccountsApi", Mock(return_value=api))

    budget_id = 123

    # Create an instance of YnabClient with mock storage and logger
    ynab_client = YnabClient(Mock())

    # Call the method under test
    ynab_client.get_account_for_budget(budget_id)

    # Assertions
    api.get_accounts.assert_called_once_with(budget_id)


def test_get_budgets_calls_api(monkeypatch):
    api = Mock(spec=ynab.BudgetsApi)
    api.get_budgets.return_value = Mock(data=Mock(budgets=[]))
    monkeypatch.setattr(ynab, "BudgetsApi", Mock(return_value=api))

    # Create an instance of YnabClient with mock storage and logger
    ynab_client = YnabClient(Mock())

    # Call the method under test
    ynab_client.get_budgets()

    # Assertions
    api.get_budgets.assert_called_once_with()


def test_get_transactions_for_account_calls_api(monkeypatch):
    api = Mock(spec=ynab.TransactionsApi)
    api.get_transactions_by_account.return_value = Mock(data=Mock(transactions=[]))
    monkeypatch.setattr(ynab, "TransactionsApi", Mock(return_value=api))

    ynab_account = Mock()
    ynab_account.budget_id = 123
    ynab_account.id = 456
    last_runmoment = Mock()
    last_runmoment.date.return_value = "2020-01-01"

    # Create an instance of YnabClient with mock storage and logger
    ynab_client = YnabClient(Mock())

    # Call the method under test
    ynab_client.get_transactions_for_account(ynab_account, last_runmoment)

    # Assertions
    api.get_transactions_by_account.assert_called_once_with(
        123, 456, since_date="2020-01-01"
    )


def test_create_transaction_calls_api(monkeypatch):
    api = Mock(spec=ynab.TransactionsApi)
    monkeypatch.setattr(ynab, "TransactionsApi", Mock(return_value=api))

    transaction = Mock()
    budget_id = 123

    # Create an instance of YnabClient with mock storage and logger
    ynab_client = YnabClient(Mock())

    # Call the method under test
    ynab_client.create_transaction(transaction, budget_id)

    # Assertions
    api.create_transaction.assert_called_once_with(
        budget_id, data={"transaction": transaction}
    )
