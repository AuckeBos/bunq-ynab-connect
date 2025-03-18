from typing import Any

import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from pydantic import Field

from bunq_ynab_connect.classification.preprocessing.features import Features
from bunq_ynab_connect.models.bunq_payment import BunqPayment


class AutoGluonFeatures(Features):
    """Extracts features that serve as input to auto gluon."""

    date_features: DatetimeFeatures | None = Field(init=False, default=None)

    def fit(self, X: list[BunqPayment], _: Any | None = None) -> pd.DataFrame:
        return self

    def transform(self, X: list[BunqPayment]) -> pd.DataFrame:
        data = [
            {
                "amount": payment.amount["value"],
                "counterparty_alias": payment.counterparty_alias_name,
                "created": payment.created,
                "description": payment.description,
            }
            for payment in X
        ]
        return pd.DataFrame(data)
