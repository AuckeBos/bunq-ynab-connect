from logging import LoggerAdapter
from typing import Iterable, List

import ynab
from kink import inject
from ynab import ApiClient

from bunq_ynab_connect.clients.ynab_client import YnabClient
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.ynab_account import YnabAccount
from bunq_ynab_connect.models.ynab.ynab_budget import YnabBudget


class YnabAccountExtractor(AbstractExtractor):
    """
    A FullLoadExtractor that extracts all accounts from YNAB.
    Is FullLoad since the accounts do not have an update date.

    Attributes:
        client: The YNAB client to use to get the accounts
        IS_FULL_LOAD: Whether the extractor is a full load extractor
    """

    client: YnabClient
    IS_FULL_LOAD = True

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, client: YnabClient
    ):
        super().__init__("ynab_accounts", storage, logger)
        self.client = client

    def get_budgets(self) -> Iterable[YnabBudget]:
        """
        Load the budgets. Assume the BudgetExtractor has already run.
        """
        budgets_dicts = self.storage.get("ynab_budgets")
        budgets = [YnabBudget(**b) for b in budgets_dicts]
        return budgets

    def load(self) -> List[dict]:
        """
        Load the data from the source.
        Loads all accounts from all budgets.
        Uses custom YnabAccount model to convert to dict: includes budget_id.
        """
        budgets = self.get_budgets()
        accounts = []
        for b in budgets:
            accounts_for_budget = self.client.get_account_for_budget(b.id)
            accounts_for_budget = [
                {"budget_id": b.id, **a.to_dict()} for a in accounts_for_budget
            ]
            accounts_for_budget = [YnabAccount(**a).dict() for a in accounts_for_budget]
            accounts.extend(accounts_for_budget)
        return accounts
