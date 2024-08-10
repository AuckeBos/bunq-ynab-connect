from __future__ import annotations

from pydantic import BaseModel


class YnabAccount(BaseModel):
    """Represents an account in a Ynab budget.

    Used because the default YnabAccount model (Account) does not have a budget_id attr.
    """

    id: str
    budget_id: str
    name: str | None
    type: str | None
    on_budget: bool | None
    closed: bool | None
    note: str | None
    balance: int | None
    cleared_balance: int | None
    uncleared_balance: int | None
    transfer_payee_id: str | None
    deleted: bool | None
