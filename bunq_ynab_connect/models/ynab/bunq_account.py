from typing import Optional

from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

from bunq_ynab_connect.helpers.general import date_to_datetime


class BunqAccount(BaseModel):
    """
    BunqAccount model.
    """

    alias: Optional[list]
    avatar: Optional[dict]
    balance: Optional[dict]
    created: Optional[str]
    currency: Optional[str]
    daily_limit: Optional[dict]
    description: Optional[str]
    display_name: Optional[str]
    id: Optional[int]
    monetary_account_profile: Optional[dict]
    overdraft_limit: Optional[dict]
    public_uuid: Optional[str]
    setting: Optional[dict]
    status: Optional[str]
    sub_status: Optional[str]
    updated: Optional[str]
    user_id: Optional[int]
    all_co_owner: Optional[list]
    savings_goal: Optional[list]
    savings_goal_progress: Optional[list]

    @property
    def iban(self) -> Optional[str]:
        """
        Get the IBAN of the account.
        It is one of the aliases of the account.
        """
        for a in self.alias:
            if a.type_ == "IBAN":
                return a.value
        return None
