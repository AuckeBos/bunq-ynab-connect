import os
from datetime import datetime
from logging import LoggerAdapter
from typing import List

import ynab
from kink import inject
from ynab import ApiClient, TransactionDetail
from ynab.models.account import Account
from ynab.models.budget_summary import BudgetSummary

from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.models.ynab_account import YnabAccount


@inject
class YnabClient:
    """
    Client for the YNAB API.

    Attributes:
        logger: The logger to use
        client: The YNAB API client
    """

    logger: LoggerAdapter
    client: ApiClient

    def __init__(self, logger: LoggerAdapter):
        self.logger = logger
        self.client = self._load_api_client()

    def _load_api_client(self) -> ApiClient:
        """
        Load the YNAB API client.
        If no token is found in the environment variables, raise an exception.
        Initialize a Configuration object with the token.
        """
        token = os.getenv("YNAB_TOKEN")
        if not token:
            self.logger.error("No ynab token found")
            raise Exception("Please set your your ynab token as YNAB_TOKEN")

        configuration = ynab.Configuration()
        configuration.api_key["Authorization"] = token
        configuration.api_key_prefix["Authorization"] = "Bearer"
        client = ApiClient(configuration)
        return client

    def get_account_for_budget(self, budget_id: str) -> List[Account]:
        """
        Load the accounts for a budget.
        """
        api = ynab.AccountsApi(self.client)
        try:
            response = api.get_accounts(budget_id)
            accounts = response.data.accounts
            self.logger.info(f"Loaded {len(accounts)} accounts for budget {budget_id}")
            return accounts
        except Exception as e:
            self.logger.error(f"Could not get accounts for budget {budget_id}: {e}")
            raise Exception(f"Could not get accounts for budget {budget_id}: {e}")

    def get_budgets(self) -> List[BudgetSummary]:
        """
        Load the budgets.
        """
        api = ynab.BudgetsApi(self.client)
        try:
            response = api.get_budgets()
            budgets = response.data.budgets
            self.logger.info(f"Loaded {len(budgets)} budgets")
            return budgets
        except Exception as e:
            self.logger.error(f"Could not get budgets: {e}")
            raise Exception(f"Could not get budgets: {e}")

    def get_transactions_for_account(
        self, account: YnabAccount, last_runmoment: datetime = None
    ) -> List[TransactionDetail]:
        """
        Load the transactions for an account.

        Args:
            account: The account to load the transactions for
            last_runmoment: Only load transactions since this date
        """
        api = ynab.TransactionsApi(self.client)
        result = api.get_transactions_by_account(
            account.budget_id, account.id, since_date=last_runmoment.date()
        ).data.transactions
        if len(result):
            self.logger.info(
                f"Loaded {len(result)} transactions for account {account.name} since {last_runmoment}"
            )
        return result

    def create_transaction(
        self, transaction: TransactionDetail, budget_id: str
    ) -> None:
        """
        Add a transaction to an account.

        Args:
            transaction: The transaction to add
            budget_id: The budget id to add the transaction to
        """
        api = ynab.TransactionsApi(self.client)
        try:
            api.create_transaction(budget_id, data={"transaction": transaction})
            self.logger.info(
                f"Added transaction {transaction.memo} to account {transaction.account_id}"
            )
        except Exception as e:
            self.logger.error(
                f"Could not add transaction {transaction} to budget {budget_id}: {e}"
            )
            raise Exception(
                f"Could not add transaction {transaction} to budget {budget_id}: {e}"
            )
