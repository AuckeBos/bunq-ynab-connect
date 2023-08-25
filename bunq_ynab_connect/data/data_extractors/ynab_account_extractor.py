from logging import LoggerAdapter
from typing import Iterable, List

import ynab
from kink import inject
from ynab import ApiClient

from bunq_ynab_connect.clients.ynab_client import YnabClient
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.ynab_budget import YnabBudget


class YnabAccountExtractor(AbstractExtractor):
    client: YnabClient
    IS_FULL_LOAD = True

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, client: YnabClient
    ):
        super().__init__("ynab_accounts", storage, logger)
        self.client = client

    def get_budgets(self) -> Iterable[YnabBudget]:
        budgets_dicts = self.storage.get("ynab_budgets")
        budgets = [YnabBudget(**b) for b in budgets_dicts]
        return budgets

    def load(self) -> List[dict]:
        budgets = self.get_budgets()
        accounts = []
        for b in budgets:
            accounts.extend(self.client.get_account_for_budget(b.id))
        accounts_dict = [a.to_dict() for a in accounts]
        return accounts_dict
