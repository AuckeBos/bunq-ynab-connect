from logging import LoggerAdapter

from kink import inject

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction


@inject
class PaymentMatcher:
    """
    Class that maps a Bunq account to a YNAB Account.
    Assume that the "notes" field of a YNAB account equals the IBAN to which it belongs
    """

    @inject
    def __init__(self, storage: AbstractStorage, logger: LoggerAdapter):
        self.storage = storage
        self.logger = logger

    def is_match(self, bunq_payment: BunqPayment, ynab_transaction: YnabTransaction):
        """
        Match when:
        - The dates are equal
        - The amounts are equal
        """
        return bunq_payment.created.date() == ynab_transaction.date.date() and float(
            bunq_payment.amount["value"]
        ) == round(ynab_transaction.amount / 1000, 2)

    def has_match(self, payment: BunqPayment):
        """
        Check if a payment has a match.
        """
        transactions = self.storage.find(
            "ynab_transactions",
            [
                ("account_id", "eq", ynab_account_id),
                ("date", "gte", self.last_runmoment),
            ],
        )
