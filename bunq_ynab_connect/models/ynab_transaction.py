from datetime import datetime

from pydantic import BaseModel, validator

from bunq_ynab_connect.helpers.general import date_to_datetime


class YnabTransaction(BaseModel):
    """Represents a transaction on an account of a budget in Ynab.

    Used because the default YnabTransaction model (TransactionDetail)
    has a property of type 'date', which is not supported by pymongo.
    """

    id: str | None
    budget_id: str | None
    date: datetime | None
    amount: int | None
    memo: str | None
    cleared: str | None
    approved: bool | None
    flag_color: str | None
    account_id: str | None
    payee_id: str | None
    category_id: str | None
    transfer_account_id: str | None
    transfer_transaction_id: str | None
    matched_transaction_id: str | None
    import_id: str | None
    deleted: bool | None
    account_name: str | None
    payee_name: str | None
    category_name: str | None
    subtransactions: list | None
    # Convert dates to datetime
    _convert_dates = validator("date", allow_reuse=True, pre=True)(date_to_datetime)
