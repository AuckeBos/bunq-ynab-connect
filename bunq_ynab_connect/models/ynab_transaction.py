from datetime import datetime
from typing import Optional

from bunq_ynab_connect.helpers.general import date_to_datetime
from pydantic import BaseModel, validator


class YnabTransaction(BaseModel):
    """
    YnabTransaction model.
    Used because the default YnabTransaction model (TransactionDetail), has a property of type 'date', which is not supported by pymongo.
    """

    id: Optional[str]
    budget_id: Optional[str]
    date: datetime
    amount: int
    memo: Optional[str]
    cleared: Optional[str]
    approved: Optional[bool]
    flag_color: Optional[str]
    account_id: Optional[str]
    payee_id: Optional[str]
    category_id: Optional[str]
    transfer_account_id: Optional[str]
    transfer_transaction_id: Optional[str]
    matched_transaction_id: Optional[str]
    import_id: Optional[str]
    deleted: Optional[bool]
    account_name: Optional[str]
    payee_name: Optional[str]
    category_name: Optional[str]
    subtransactions: Optional[list]
    # Convert dates to datetime
    _convert_dates = validator("date", allow_reuse=True, pre=True)(date_to_datetime)
