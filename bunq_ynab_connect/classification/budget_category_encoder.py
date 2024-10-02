import pandas as pd
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

    def fit(self, y: pd.DataFrame) -> "BudgetCategoryEncoder":
        self.id_to_name = y.set_index("category_id")["category_name"].to_dict()
        self.encoder.fit(y["category_id"])
        return self

    def transform(self, y: pd.DataFrame) -> list[int]:
        return self.encoder.transform(y["category_id"])

    def inverse_transform(self, y: pd.DataFrame) -> list[str]:
        return self.encoder.inverse_transform(y)
