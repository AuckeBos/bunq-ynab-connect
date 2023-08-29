from typing import Optional

from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

from bunq_ynab_connect.helpers.general import date_to_datetime


class BunqPayment(BaseModel):
    """
    BunqPayment model.
    """

    id: Optional[int]
    alias: Optional[dict]
    amount: Optional[dict]
    attachment: Optional[list]
    balance_after_mutation: Optional[dict]
    counterparty_alias: Optional[dict]
    created: Optional[str]
    description: Optional[str]
    monetary_account_id: Optional[int]
    request_reference_split_the_bill: Optional[list]
    sub_type: Optional[str]
    type: Optional[str]
    updated: Optional[str]
