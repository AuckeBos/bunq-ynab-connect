from datetime import datetime
from typing import Optional

from pydantic import BaseModel, validator
from pydantic.dataclasses import dataclass

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
