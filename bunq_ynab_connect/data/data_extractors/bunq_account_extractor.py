from __future__ import annotations

import json
from typing import TYPE_CHECKING

from kink import inject

from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor

if TYPE_CHECKING:
    from logging import LoggerAdapter

    from bunq_ynab_connect.clients.bunq_client import BunqClient
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
        accounts = self.client.get_accounts()
        return [json.loads(a.to_json()) for a in accounts]
