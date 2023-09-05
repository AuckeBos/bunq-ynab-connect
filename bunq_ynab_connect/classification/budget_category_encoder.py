from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from bunq_ynab_connect.models.ynab.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class BudgetCategoryEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder for the budget categories.
    Consumes a list of YnabTransactions and returns a list of integers.
    """

    encoder: LabelEncoder

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, y: List[YnabTransaction]):
        categories = [transaction.category_name for transaction in y]
        self.encoder.fit(categories)
        return self

    def transform(self, y: List[YnabTransaction]):
        categories = [transaction.category_name for transaction in y]
        return self.encoder.transform(categories)

    def inverse_transform(self, y):
        return self.encoder.inverse_transform(y)
