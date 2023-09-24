from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

from bunq_ynab_connect.helpers.general import date_to_datetime
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction


class MatchedTransaction(BaseModel):
    match_id: str
    bunq_payment: BunqPayment
    ynab_transaction: YnabTransaction
