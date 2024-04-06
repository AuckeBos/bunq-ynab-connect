from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class BunqPayment(BaseModel):
    """
    BunqPayment model.
    """

    id: Optional[int]
    alias: Optional[dict]
    amount: dict
    attachment: Optional[list]
    balance_after_mutation: Optional[dict]
    counterparty_alias: Optional[dict]
    created: datetime
    description: Optional[str]
    monetary_account_id: Optional[int]
    request_reference_split_the_bill: Optional[list]
    sub_type: Optional[str]
    type: Optional[str]
    updated: Optional[datetime]
