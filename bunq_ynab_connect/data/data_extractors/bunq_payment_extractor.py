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


class BunqPaymentExtractor(AbstractExtractor):
    """
    Extractor for bunq payments.
    Loads all payments from all accounts.

    Attributes:
        PAYMENTS_PER_PAGE: The amount of payments to load per page
        After each page, the last payment is checked to see if it is older than the last runmoment
        If it is older, we are done loading payments
    """

    client: BunqClient
    PAYMENTS_PER_PAGE = 10

    @inject
    def __init__(
        self, storage: AbstractStorage, logger: LoggerAdapter, client: BunqClient
    ) -> None:
        super().__init__("bunq_payments", storage, logger)
        self.client = client

    def load(self) -> List:
        """
        Load the data from the source.
        Loads all payments from all accounts
        """
        accounts = self.client.get_accounts()
        payments = []
        for account in accounts:
            payments.extend(
                self.client.get_payments_for_account(account, self.last_runmoment)
            )
        return payments
