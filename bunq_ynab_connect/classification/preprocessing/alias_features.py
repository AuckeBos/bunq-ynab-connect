import pandas as pd
from feature_engine.encoding import StringSimilarityEncoder

from bunq_ynab_connect.classification.preprocessing.features import Features
from bunq_ynab_connect.models.bunq_payment import BunqPayment


class AliasFeatures(Features):
    """Extract features from the aliasses.

    Use StringSimilarityEncoder, to allow the model to find similarities based on the
    alias.
    """

    alias_features: StringSimilarityEncoder | None

    top_categories: int | None
    enabled: bool

    def __init__(self, top_categories: int | None = None, enabled: bool = True):
        self.top_categories = top_categories
        self.enabled = enabled

    def fit(self, X: list[BunqPayment], _) -> "AliasFeatures":
        if not self.enabled:
            return self
        self.alias_features = StringSimilarityEncoder(
            top_categories=self.top_categories
        )
        self.alias_features.fit(self.payments_to_aliases(X))
        return self

    def transform(self, X: list[BunqPayment]) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame(index=range(len(X)))
        features = self.alias_features.transform(self.payments_to_aliases(X))
        return pd.DataFrame(
            features, columns=self.alias_features.get_feature_names_out()
        )

    def payments_to_aliases(self, X: list[BunqPayment]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "alias": [payment.alias["display_name"] for payment in X],
                "counterparty_alias": [
                    payment.counterparty_alias["display_name"] for payment in X
                ],
            }
        )
