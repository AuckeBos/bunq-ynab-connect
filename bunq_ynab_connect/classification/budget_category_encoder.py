from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LabelEncoder


class BudgetCategoryEncoder(BaseEstimator, TransformerMixin):
    """Encoder for the budget categories.

    Consumes a list of YnabTransactions and returns a list of integers.

    Attributes
    ----------
        encoder: LabelEncoder used to encode the categories
        id_to_name: dict[str, str]
            Maps IDs to names. To be able to log the category names later on.

    """

    encoder: LabelEncoder
    id_to_name: dict[str, str]

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, y: list[dict]) -> "BudgetCategoryEncoder":
        categories = [transaction["category_id"] for transaction in y]
        self.id_to_name = {
            transaction["category_id"]: transaction["category_name"]
            for transaction in y
        }
        self.encoder.fit(categories)
        return self

    def transform(self, y: list[dict]) -> list[int]:
        categories = [transaction["category_id"] for transaction in y]
        return self.encoder.transform(categories)

    def inverse_transform(self, y: Any) -> list[str]:
        return self.encoder.inverse_transform(y)
