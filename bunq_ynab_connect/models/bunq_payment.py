from datetime import datetime

from pydantic import BaseModel
from sqlmodel import Field, SQLModel

from bunq_ynab_connect.models.table import Table


class Amount(SQLModel, table=True):
    currency: str
    value: str


# todo: fix issue with everything needing to be a table


class BunqPaymentBase(BaseModel):
    """Represents a payment in a BunqAccount."""

    id: int | None = Field(primary_key=True, default=None)
    alias: dict | None
    amount: dict | None
    attachment: list | None
    counterparty_alias: dict | None
    created: datetime | None
    description: str | None
    monetary_account_id: int | None
    sub_type: str | None
    type: str | None
    updated: datetime | None


class BunqPaymentSchema(Table, table=True):
    """Represents a payment in a BunqAccount."""

    id: int | None
    alias: dict | None
    amount: dict | None
    attachment: list | None
    counterparty_alias: dict | None
    created: datetime | None
    description: str | None
    monetary_account_id: int | None
    sub_type: str | None
    type: str | None
    updated: datetime | None


class BunqPayment(Table, table=True):
    """Represents a payment in a BunqAccount."""

    id: int | None
    amount: dict | None
    attachment: list | None
    counterparty_alias: dict | None
    created: datetime | None
    description: str | None
    monetary_account_id: int | None
    sub_type: str | None
    type: str | None
    updated: datetime | None
