from datetime import datetime

from pydantic import BaseModel


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

    @property
    def alias_name(self) -> str:
        return self._get_name(self.alias)

    @property
    def counterparty_alias_name(self) -> str:
        return self._get_name(self.counterparty_alias)

    @staticmethod
    def _get_name(alias: dict) -> str:
        """Get the name of the alias.

        Handle missing name w/o errors.
        """
        if "display_name" in alias:
            return alias["display_name"]
        if "name" in alias:
            return alias["name"]
        return "Unknown"
