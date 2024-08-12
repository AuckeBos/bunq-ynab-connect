import os
from datetime import datetime
from logging import LoggerAdapter

import ynab
from kink import inject
from ynab import ApiClient, TransactionDetail
from ynab.models.account import Account
from ynab.models.budget_summary import BudgetSummary

from bunq_ynab_connect.models.ynab_account import YnabAccount


@inject
class YnabClient:
    """Client for the YNAB API.

    Attributes
    ----------
        logger: The logger to use
        client: The YNAB API client

    """

    logger: LoggerAdapter
    client: ApiClient

    def __init__(self, logger: LoggerAdapter):
        self.logger = logger
        self.client = self._load_api_client()

    def _load_api_client(self) -> ApiClient:
        """Load the YNAB API client.

        If no token is found in the environment variables, raise an exception.
        Initialize a Configuration object with the token.
        """
        token = os.getenv("YNAB_TOKEN")
        if not token:
            msg = "Please set your your ynab token as YNAB_TOKEN"
            self.logger.error(msg)
            raise ValueError(msg)

        configuration = ynab.Configuration()
        configuration.api_key["Authorization"] = token
        configuration.api_key_prefix["Authorization"] = "Bearer"
        return ApiClient(configuration)

    def get_account_for_budget(self, budget_id: str) -> list[Account]:
        """Load the accounts for a budget."""
        api = ynab.AccountsApi(self.client)
        try:
            response = api.get_accounts(budget_id)
            accounts = response.data.accounts
            self.logger.info(
                "Loaded %s accounts for budget %s", len(accounts), budget_id
            )
        except Exception as e:
            msg = f"Could not get accounts for budget {budget_id}"
            self.logger.exception(msg)
            raise OSError(msg) from e
        else:
            return accounts

    def get_budgets(self) -> list[BudgetSummary]:
        api = ynab.BudgetsApi(self.client)
        try:
            response = api.get_budgets()
            budgets = response.data.budgets
            self.logger.info("Loaded %s budgets", len(budgets))
        except Exception as e:
            msg = "Could not get budgets"
            self.logger.exception(msg)
            raise OSError(msg) from e
        else:
            return budgets

    def get_transactions_for_account(
        self, account: YnabAccount, last_runmoment: datetime | None = None
    ) -> list[TransactionDetail]:
        """Load the transactions for an account.

        Only load transactions since the last runmoment.
        """
        api = ynab.TransactionsApi(self.client)
        result = api.get_transactions_by_account(
            account.budget_id, account.id, since_date=last_runmoment.date()
        ).data.transactions
        if len(result):
            self.logger.info(
                "Loaded %s transactions for account %s", len(result), account.name
            )
        return result

    def create_transaction(
        self, transaction: TransactionDetail, budget_id: str
    ) -> None:
        """Add a transaction to a budget."""
        api = ynab.TransactionsApi(self.client)
        try:
            api.create_transaction(budget_id, data={"transaction": transaction})
            self.logger.info(
                "Added transaction %s to budget %s", transaction.memo, budget_id
            )
        except Exception as e:
            msg = f"Could not add transaction {transaction} to budget {budget_id}"
            self.logger.exception(msg)
            raise OSError(msg) from e
