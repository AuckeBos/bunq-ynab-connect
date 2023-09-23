from logging import LoggerAdapter
from typing import List

import ynab
from kink import inject
from ynab import ApiClient

from bunq_ynab_connect.clients.ynab_client import YnabClient
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab_budget import YnabBudget


class YnabBudgetExtractor(AbstractExtractor):
    """
    Extractor for YNAB budgets.
    """

    client: YnabClient

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, client: YnabClient
    ):
        super().__init__("ynab_budgets", storage, logger)
        self.client = client

    def load(self) -> List[dict]:
        """
        Use the YNAB client to get the budgets.
        Use custom YnabBudget model to convert to dict.
        """
        budgets = self.client.get_budgets()

        budgets = [YnabBudget(**b.to_dict()) for b in budgets]
        budgets_dict = [b.dict() for b in budgets]
        return budgets_dict
