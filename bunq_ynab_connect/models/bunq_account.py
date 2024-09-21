from typing import Optional

from kink import inject
from pydantic import BaseModel
from sqlmodel import Field, Relationship, SQLModel

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.table import Table


class BunqAlias(SQLModel, table=True):
    id: int | None = Field(primary_key=True, default=None)
    name: str
    type: str
    value: str
    bunq_account_id: int | None = Field(default=None, foreign_key="bunqaccount.id")
    bunq_accounts: list["BunqAccount"] = Relationship(back_populates="aliasses")


class BunqAccountBase(BaseModel):
    id: int | None = Field(primary_key=True, default=None)
    created: str | None
    currency: str | None
    description: str | None
    display_name: str | None
    public_uuid: str | None
    status: str | None
    sub_status: str | None
    updated: str | None
    user_id: int | None


class BunqAccountSchema(BunqAccountBase):
    from pydantic import Field

    aliasses: list[BunqAlias] | None = Field(alias="alias", default=None)


class BunqAccount(BunqAccountBase, Table, table=True):
    """Represent an account in Bunq."""

    aliasses: list[BunqAlias] = Relationship(back_populates="bunq_accounts")

    @property
    def iban(self) -> str | None:
        """Get the IBAN of the account.

        It is one of the aliases of the account.
        """
        for a in self.aliasses:
            if a.type == "IBAN":
                return a.value
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
