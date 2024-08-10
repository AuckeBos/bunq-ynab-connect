from datetime import datetime
from typing import List, Optional

from kink import inject
from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.general import date_to_datetime


class YnabBudget(BaseModel):
    """
    YnabBudget model.
    Used because the default YnabBudget model (BudgetSummary), has first and last month as date, which is not supported by pymongo.
    """

    id: Optional[str]
    name: Optional[str]
    last_modified_on: Optional[datetime]
    first_month: Optional[datetime]
    last_month: Optional[datetime]
    date_format: Optional[dict]
    currency_format: Optional[dict]
    # Convert dates to datetime
    _convert_dates = validator("first_month", "last_month", allow_reuse=True, pre=True)(
        date_to_datetime
    )

    @staticmethod
    def get_budget_ids(storage: AbstractStorage) -> List[str]:
        """
        Get all budget ids from the storage.
        """
        ynab_budgets = storage.find("ynab_budgets")
        return [budget["id"] for budget in ynab_budgets]
