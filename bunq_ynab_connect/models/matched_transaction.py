from pydantic import BaseModel

from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction


class MatchedTransaction(BaseModel):
    """Represents a matched transaction between a BunqPayment and a YnabTransaction."""

    match_id: str
    bunq_payment: BunqPayment
    ynab_transaction: YnabTransaction
