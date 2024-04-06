from typing import Dict, List, Union

from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.clients.ynab_client import YnabClient
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction
from kink import inject


@inject
class PaymentTransactionMapper:
    """
    Converts a list of payments and a list of transactions
    into a list of matches. A match is a tuple of a payment and a transaction.
    """

    MINIMUM_AMOUNT_FOR_SANE_TRANSACTION = 0.05

    @inject
    def __init__(self, bunq_client: BunqClient, ynab_client: YnabClient):
        self.bunq_client = bunq_client
        self.ynab_client = ynab_client

    def map(
        self, payments: List[BunqPayment], transactions: List[YnabTransaction]
    ) -> List[Dict[str, Union[int, Dict]]]:
        matches = []
        for transaction in transactions:
            if not self.sanity_check_transaction(transaction):
                continue
            for payment in payments:
                if not self.sanity_check_payment(payment):
                    continue
                if self.is_match(payment, transaction):
                    matches.append(
                        {
                            "match_id": transaction.id,
                            "bunq_payment": payment.dict(),
                            "ynab_transaction": transaction.dict(),
                        }
                    )
                    break
        return matches

    def sanity_check_transaction(self, transaction: YnabTransaction) -> bool:
        """
        Sanity check for whether a ynab transaction should be included the map.
        Check that the amount is at least +=0.05. Lower amounts are test payments
        """
        if not abs(transaction.amount) > self.MINIMUM_AMOUNT_FOR_SANE_TRANSACTION:
            return False
        return True

    def sanity_check_payment(self, payment: BunqPayment) -> bool:
        """
        Sanity check for whether a payment should be included in the map.
        - Check that the amount is at least +=0.05. Lower amounts are test payments
        """
        if (
            not abs(float(payment.amount["value"]))
            > self.MINIMUM_AMOUNT_FOR_SANE_TRANSACTION
        ):
            return False
        return True

    def is_match(self, payment: BunqPayment, transaction: YnabTransaction) -> bool:
        """
        Match when:
        - The dates are equal
        - The amounts are equal
        """
        return payment.created.date() == transaction.date.date() and float(
            payment.amount["value"]
        ) == round(transaction.amount / 1000, 2)

    def payment_yet_exists(self, payment: BunqPayment) -> bool:
        """
        Check if a payment already exists in the ynab_transaction table.
        If so, it should not be included in the map (to prevent double imports).
        """
        pass
