import pandas as pd
from feature_engine.encoding import StringSimilarityEncoder
from pydantic import Field

from bunq_ynab_connect.classification.preprocessing.features import Features
from bunq_ynab_connect.models.bunq_payment import BunqPayment


class CounterpartySimilarityFeatures(Features):
    """Extract features from the aliasses.

    Use StringSimilarityEncoder, to allow the model to find similarities based on the
    alias.
    """

    alias_features: StringSimilarityEncoder | None

    def fit(self, X: list[BunqPayment]) -> pd.DataFrame:
        self.alias_features = StringSimilarityEncoder()
        self.alias_features.fit(self.payments_to_aliases(X))

    def transform(self, X: list[BunqPayment]) -> pd.DataFrame:
        features = self.alias_features.transform(self.payments_to_aliases(X))
        return pd.DataFrame(
            features, columns=self.alias_features.get_feature_names_out()
        )

    def payments_to_aliases(self, X: list[BunqPayment]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "counterparty_alias": [
                    payment.counterparty_alias["display_name"] for payment in X
                ],
            }
        )
