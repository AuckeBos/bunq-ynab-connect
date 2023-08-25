import os
from logging import LoggerAdapter
from typing import List

import ynab
from kink import inject
from ynab import ApiClient
from ynab.models.account import Account
from ynab.models.budget_summary import BudgetSummary

from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor


@inject
class YnabClient:
    """
    Client for the YNAB API.
    """

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
            return budgets
        except Exception as e:
            self.logger.error(f"Could not get budgets: {e}")
            raise Exception(f"Could not get budgets: {e}")