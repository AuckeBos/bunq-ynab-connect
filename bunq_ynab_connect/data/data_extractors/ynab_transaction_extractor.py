from logging import LoggerAdapter
from typing import List

import ynab
from kink import inject
from pyparsing import Iterable
from ynab import ApiClient

from bunq_ynab_connect.clients.ynab_client import YnabClient
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.ynab_account import YnabAccount
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class YnabTransactionExtractor(AbstractExtractor):
    """
    Extractor for YNAB transactions.
    """

    client: YnabClient

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, client: YnabClient
    ):
        super().__init__("ynab_transactions", storage, logger)
        self.client = client

    def load(self) -> List[dict]:
        """
        Use the YNAB client to get the transactions.
        Use custom YnabTransaction model to convert to dict, to convert dates to datetime.
        """
        accounts = self.storage.get_as_entity("ynab_accounts", YnabAccount, False)
        transactions = []
        for a in accounts:
            transactions_for_account = self.client.get_transactions_for_account(
                a, self.last_runmoment
            )
            transactions_for_account = [
                YnabTransaction(**a.to_dict()).dict() for a in transactions_for_account
            ]
            transactions.extend(transactions_for_account)
        return transactions
