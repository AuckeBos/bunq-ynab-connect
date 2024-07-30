from __future__ import annotations

from typing import TYPE_CHECKING

from kink import inject

from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.models.ynab_account import YnabAccount
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction

if TYPE_CHECKING:
    from logging import LoggerAdapter

    from bunq_ynab_connect.clients.ynab_client import YnabClient
    from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class YnabTransactionExtractor(AbstractExtractor):
    """Extractor for YNAB transactions."""

    client: YnabClient

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, client: YnabClient
    ):
        super().__init__("ynab_transactions", storage, logger)
        self.client = client

    def load(self) -> list[dict]:
        """Use the YNAB client to get the transactions."""
        accounts = self.storage.get_as_entity(
            "ynab_accounts", YnabAccount, provide_kwargs_as_json=False
        )
        transactions = []
        for a in accounts:
            transactions_for_account = self.client.get_transactions_for_account(
                a, self.last_runmoment
            )
            transactions_for_account = [
                YnabTransaction(
                    **t.to_dict(),
                    budget_id=a.budget_id,
                ).dict()
                for t in transactions_for_account
            ]
            transactions.extend(transactions_for_account)
        return transactions
