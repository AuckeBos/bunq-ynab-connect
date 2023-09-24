from typing import Optional

from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

from bunq_ynab_connect.helpers.general import date_to_datetime


class YnabAccount(BaseModel):
    """
    YnabAccount model.
    Used because the default YnabAccount model (Account) does not have the budget_id included.
    """

    id: str
    budget_id: str
    name: Optional[str]
    type: Optional[str]
    on_budget: Optional[bool]
    closed: Optional[bool]
    note: Optional[str]
    balance: Optional[int]
    cleared_balance: Optional[int]
    uncleared_balance: Optional[int]
    transfer_payee_id: Optional[str]
    deleted: Optional[bool]
