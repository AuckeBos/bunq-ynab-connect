from typing import Optional

from kink import inject
from pydantic import BaseModel

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class BunqAccount(BaseModel):
    """Represent an account in Bunq."""

    alias: list | None
    avatar: dict | None
    balance: dict | None
    created: str | None
    currency: str | None
    daily_limit: dict | None
    description: str | None
    display_name: str | None
    id: int | None
    monetary_account_profile: dict | None
    public_uuid: str | None
    setting: dict | None
    status: str | None
    sub_status: str | None
    updated: str | None
    user_id: int | None

    @property
    def iban(self) -> str | None:
        """Get the IBAN of the account.

        It is one of the aliases of the account.
        """
        for a in self.alias:
            if a["type"] == "IBAN":
                return a["value"]
        return None

    @staticmethod
    @inject
    def by_iban(storage: AbstractStorage, iban: str) -> Optional["BunqAccount"]:
        """Get a Bunq account by IBAN."""
        accounts = storage.get_as_entity(
            "bunq_accounts", BunqAccount, provide_kwargs_as_json=False
        )
        for account in accounts:
            if account.iban == iban:
                return account
        return None
