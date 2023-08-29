from ast import Dict
from logging import LoggerAdapter

from bunq.sdk.model.generated.endpoint import (
    BunqResponsePaymentList,
    MonetaryAccountBank,
    Payment,
)
from kink import inject
from ynab import TransactionDetail

from bunq_ynab_connect.clients.ynab_client import YnabClient
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.ynab_account import YnabAccount
from bunq_ynab_connect.sync_bunq_to_ynab.bunq_account_to_ynab_account_mapper import (
    BunqAccountToYnabAccountMapper,
)
from bunq_ynab_connect.sync_bunq_to_ynab.payment_queue import PaymentQueue


class PaymentSyncer:
    """
    Class that syncs payments from Bunq to Ynab.
    """

    FLAG_COLOR = "blue"
    CLEARING_STATUS = "uncleared"

    logger: LoggerAdapter
    storage: AbstractStorage
    client: YnabClient
    mapper: BunqAccountToYnabAccountMapper
    queue: PaymentQueue
    account_map: Dict[str, YnabAccount]

    @inject
    def __init__(
        self,
        logger: LoggerAdapter,
        storage: AbstractStorage,
        client: YnabClient,
        mapper: BunqAccountToYnabAccountMapper,
        queue: PaymentQueue,
    ):
        self.logger = logger
        self.storage = storage
        self.client = client
        self.mapper = mapper
        self.queue = queue
        self.account_map = {}  # Set in sync

    def payment_to_transction(
        self, payment: Payment, account: YnabAccount
    ) -> TransactionDetail:
        """
        Create a YNAB transaction from a Bunq payment.
        Use values from the payment to fill in the transaction.

        Args:
            payment: The Bunq payment
            account: The YNAB account to which the payment belongs

        Returns:
            The YNAB transaction to be created
        """
        transaction = TransactionDetail(
            account_id=account.id,
            amount=payment.amount.value,
            currency=payment.amount.currency,
            date=payment.created,
            payee_name=payment.counterparty_alias.display_name,
            memo=payment.description,
            cleared=self.CLEARING_STATUS,
            flag_color=self.FLAG_COLOR,
            approved=False,
        )
        return transaction

    def sync_payment(self, payment_id: str):
        """
        Sync a payment from Bunq to YNAB.
        - Load the payment from the database
        - Find the corresponding YNAB account
        - Create a transaction.

        Args:
            payment_id: The id of the payment to sync

        """
        payment = self.storage.find_one("payments", {"id": payment_id})
        if payment is None:
            raise ValueError(f"Could not find payment with id {payment_id}")
        payment = self.storage.rows_to_entities([payment], Payment)[0]
        account_id = payment.monetary_account_id
        if account_id not in self.account_map:
            self.logger.warning(
                f"Could not find YNAB account for Bunq account {account_id}. Not syncing payment {payment_id}"
            )
            return
        ynab_account: YnabAccount = self.account_map[account_id]
        transaction = self.payment_to_transaction(payment, ynab_account)
        self.client.create_transaction(transaction, ynab_account.budget_id)

    def sync(self):
        """
        Sync all payments in the queue from Bunq to YNAB.
        """
        self.account_map = self.mapper.map()
        while self.queue:
            with self.queue.pop() as payment_id:
                self.sync_payment(payment_id, map)
