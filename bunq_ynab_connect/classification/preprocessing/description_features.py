import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from bunq_ynab_connect.classification.preprocessing.features import Features
from bunq_ynab_connect.models.bunq_payment import BunqPayment


class DescriptionFeatures(Features):
    """Extract TF-IDF features from the description of the payments."""

    tfidf_features: TfidfVectorizer | None
    max_features: int
    enabled: bool

    def __init__(self, max_features: int | None = None, enabled: bool = True):
        self.max_features = max_features
        self.enabled = enabled

    def fit(self, X: list[BunqPayment], _) -> pd.DataFrame:
        if not self.enabled:
            return self
        self.fit_tfidf_features(X)
        return self

    def transform(self, X: list[BunqPayment]) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame(index=range(len(X)))
        features = self.tfidf_features.transform([payment.description for payment in X])
        return pd.DataFrame(
            features.todense(), columns=self.tfidf_features.get_feature_names_out()
        )

    def fit_tfidf_features(self, X: list[BunqPayment]) -> None:
        self.tfidf_features = TfidfVectorizer(
            strip_accents="ascii",
            lowercase=True,
            token_pattern=r"\b[^\d\W]+",  # noqa: S106
            max_features=self.max_features,
        )
        self.tfidf_features.fit([payment.description for payment in X])
