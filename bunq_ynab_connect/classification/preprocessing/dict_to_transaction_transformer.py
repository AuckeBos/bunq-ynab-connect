from typing import Any

from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin

from bunq_ynab_connect.models.bunq_payment import BunqPayment


class DictToTransactionTransformer(BaseEstimator, TransformerMixin):
    """Simple transformer to convert a list of dictionaries to a list of BunqPayments.

    The Features classes expect a list of BunqPayments as input. This transformer
    converts a list of dictionaries to a list of BunqPayments. It should be the first
    step in a pipeline.
    """

    def fit(self, _: Any, __: Any = None) -> "DictToTransactionTransformer":
        """No fitting is required."""
        return self

    def transform(self, X: ndarray, _: Any = None) -> list[BunqPayment]:
        """Convert the list of dictionaries to a list of BunqPayments."""
        return [BunqPayment(**x) for x in X]
