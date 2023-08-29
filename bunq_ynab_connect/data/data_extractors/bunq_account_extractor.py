import json
from logging import LoggerAdapter
from typing import List

from bunq import ApiEnvironmentType, Pagination
from bunq.sdk.context.api_context import ApiContext
from bunq.sdk.context.bunq_context import BunqContext
from bunq.sdk.model.generated import endpoint
from bunq.sdk.model.generated.endpoint import BunqResponsePaymentList, Payment
from dateutil.parser import parse
from kink import inject

from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.config import BUNQ_CONFIG_FILE
from bunq_ynab_connect.helpers.general import cache, get_public_ip


class BunqAccountExtractor(AbstractExtractor):
    """
    Extractor for Bunq accounts

    Attributes:
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

    def load(self) -> List:
        """
        Load all accounts
        """
        accounts = self.client.get_accounts()
        accounts_dicts = [json.loads(a.to_json()) for a in accounts]
        return accounts_dicts
