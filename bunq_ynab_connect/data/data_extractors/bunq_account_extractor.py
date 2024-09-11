from logging import LoggerAdapter

from kink import inject

from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class BunqAccountExtractor(AbstractExtractor):
    """Extractor for Bunq accounts.

    Attributes
    ----------
        client: The bunq client to use to get the payments
        IS_FULL_LOAD: Always load all accounts

    """

    client: BunqClient
    IS_FULL_LOAD = True

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, client: BunqClient
    ) -> None:
        super().__init__("bunq_accounts", storage, logger)
        self.client = client

    def load(self) -> list[dict]:
        return self.client.get_accounts()
