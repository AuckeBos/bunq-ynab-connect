from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from datetime import datetime


class BunqPayment(BaseModel):
    """Represents a payment in a BunqAccount."""

    id: int | None
    alias: dict | None
    amount: dict | None
    attachment: list | None
    balance_after_mutation: dict | None
    counterparty_alias: dict | None
    created: datetime | None
    description: str | None
    monetary_account_id: int | None
    request_reference_split_the_bill: list | None
    sub_type: str | None
    type: str | None
    updated: datetime | None
