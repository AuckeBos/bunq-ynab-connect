from logging import LoggerAdapter

from kink import inject

from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.bunq_account import BunqAccount
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.sync_bunq_to_ynab.payment_queue import PaymentQueue


class BunqPaymentExtractor(AbstractExtractor):
    """Extractor for bunq payments.

    Loads all payments from all accounts.

    Attributes
    ----------
        client: The bunq client to use to get the payments
        payment_queue: The payment queue to use to queue payments
            All loaded payments are added to the queue, such that they can be processed

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

    def load(self) -> list[BunqPayment]:
        """Load the data from the source.

        Loads all payments from all accounts
        """
        accounts = self.storage.get_as_entity(
            "bunq_accounts", BunqAccount, provide_kwargs_as_json=False
        )
        payments = []
        for account in accounts:
            payments.extend(
                self.client.get_payments_for_account(account, self.last_runmoment)
            )
        for payment in payments:
            self.payment_queue.add(payment["id"])
        return payments
