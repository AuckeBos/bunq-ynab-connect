from datetime import datetime

from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

from bunq_ynab_connect.helpers.general import date_to_datetime


class YnabBudget(BaseModel):
    """
    YnabBudget model.
    Used because the default YnabBudget model (BudgetSummary), has first and last month as date, which is not supported by pymongo.
    """

    id: str | None
    name: str | None
    last_modified_on: datetime | None
    first_month: datetime | None
    last_month: datetime | None
    date_format: dict | None
    currency_format: dict | None
    # Convert dates to datetime
    _convert_dates = validator("first_month", "last_month", allow_reuse=True, pre=True)(
        date_to_datetime
    )
