from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LabelEncoder


class BudgetCategoryEncoder(BaseEstimator, TransformerMixin):
    """Encoder for the budget categories.

    Consumes a list of YnabTransactions and returns a list of integers.
    """

    encoder: LabelEncoder

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, y: list[dict]) -> "BudgetCategoryEncoder":
        categories = [transaction["category_name"] for transaction in y]
        self.encoder.fit(categories)
        return self

    def transform(self, y: list[dict]) -> list[int]:
        categories = [transaction["category_name"] for transaction in y]
        return self.encoder.transform(categories)

    def inverse_transform(self, y: Any) -> list[str]:  # noqa: ANN401
        return self.encoder.inverse_transform(y)
