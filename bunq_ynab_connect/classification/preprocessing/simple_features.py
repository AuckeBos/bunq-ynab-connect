from typing import Any

import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from pydantic import Field

from bunq_ynab_connect.classification.preprocessing.features import Features
from bunq_ynab_connect.models.bunq_payment import BunqPayment


class SimpleFeatures(Features):
    """Extracts simple features from the payments.

    The features extracted are:
    - amount
    - date features.
    """

    date_features: DatetimeFeatures | None = Field(init=False, default=None)

    def fit(self, X: list[BunqPayment], _: Any) -> pd.DataFrame:
        self.fit_date_features(X)
        return self

    def transform(self, X: list[BunqPayment]) -> pd.DataFrame:
        features = self.date_features.transform(
            pd.DataFrame({"created": [payment.created for payment in X]})
        )
        features["amount"] = [payment.amount["value"] for payment in X]
        return features

    def fit_date_features(self, X: list[BunqPayment]) -> None:
        self.date_features = DatetimeFeatures(
            features_to_extract=[
                "month",
                "quarter",
                "year",
                "day_of_week",
                "weekend",
                "month_start",
                "month_end",
                "hour",
                "minute",
                "second",
            ],
            variables=["created"],
            drop_original=True,
        )
        self.date_features.fit(
            pd.DataFrame({"created": [payment.created for payment in X]})
        )

    def extract_date_features(self, payments: list[BunqPayment]) -> pd.DataFrame:
        return self.date_features.transform(
            pd.DataFrame({"created": [payment.created for payment in payments]})
        )
