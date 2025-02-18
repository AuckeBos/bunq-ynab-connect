from abc import ABC, abstractmethod

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

from bunq_ynab_connect.models.bunq_payment import BunqPayment


class Features(ABC, BaseEstimator, TransformerMixin):
    """A Base class to extract features from the bunq payments."""

    @abstractmethod
    def fit(self, X: list[BunqPayment], y=None) -> "Features": ...  # noqa: ANN001

    @abstractmethod
    def transform(self, X: list[BunqPayment]) -> DataFrame: ...
