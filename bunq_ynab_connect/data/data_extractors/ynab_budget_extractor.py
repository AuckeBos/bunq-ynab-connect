from __future__ import annotations

from typing import TYPE_CHECKING

from kink import inject

from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.models.ynab_budget import YnabBudget

if TYPE_CHECKING:
    from logging import LoggerAdapter

    from bunq_ynab_connect.clients.ynab_client import YnabClient
    from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class YnabBudgetExtractor(AbstractExtractor):
    """Extractor for YNAB budgets."""

    client: YnabClient

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, client: YnabClient
    ):
        super().__init__("ynab_budgets", storage, logger)
        self.client = client

    def load(self) -> list[dict]:
        """Use the YNAB client to get the budgets.

        Use custom YnabBudget model to convert to dict.
        """
        budgets = self.client.get_budgets()

        budgets = [YnabBudget(**b.to_dict()) for b in budgets]
        return [b.dict() for b in budgets]
