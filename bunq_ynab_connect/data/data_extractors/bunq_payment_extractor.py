import json
from logging import LoggerAdapter
from typing import List

from bunq import ApiEnvironmentType, Pagination
from bunq.sdk.context.api_context import ApiContext
from bunq.sdk.context.bunq_context import BunqContext
from bunq.sdk.model.generated import endpoint
from bunq.sdk.model.generated.endpoint import (
    BunqResponsePaymentList,
    MonetaryAccountBank,
    Payment,
)
from dateutil.parser import parse
from kink import inject

from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.config import BUNQ_CONFIG_FILE
from bunq_ynab_connect.helpers.general import cache, get_public_ip
from bunq_ynab_connect.sync_bunq_to_ynab.payment_queue import PaymentQueue


class BunqPaymentExtractor(AbstractExtractor):
    """
    Extractor for bunq payments.
    Loads all payments from all accounts.

    Attributes:
        client: The bunq client to use to get the payments
        payment_queue: The payment queue to use to queue payments
            All loaded payments are added to the queue, such that they can be processed later
    """

    client: BunqClient
    payment_queue: PaymentQueue

    @inject
    def __init__(
        self,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        client: BunqClient,
        payment_queue: PaymentQueue,
    ) -> None:
        super().__init__("bunq_payments", storage, logger)
        self.client = client
        self.payment_queue = payment_queue

    def load(self) -> List:
        """
        Load the data from the source.
        Loads all payments from all accounts
        """
        accounts = self.storage.get_as_entity(
            "bunq_accounts", MonetaryAccountBank.from_json, True
        )
        payments = []
        for account in accounts:
            payments.extend(
                self.client.get_payments_for_account(account, self.last_runmoment)
            )
        for payment in payments:
            self.payment_queue.add(payment.id_)
        payments_dict = [json.loads(pay.to_json()) for pay in payments]
        return payments_dict
