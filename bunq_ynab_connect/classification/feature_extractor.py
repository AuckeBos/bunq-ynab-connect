import re
from typing import ClassVar

import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder

from bunq_ynab_connect.models.bunq_payment import BunqPayment


class ReshapeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(X).reshape(-1, 1)


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from the bunq payments."""

    description_encoder: TfidfVectorizer
    alias_encoder: OneHotEncoder

    COLUMNS: ClassVar[list[str]] = [
        "description",
        "alias",
        "counterparty_alias",
        "amount",
        "hour",
        "minute",
        "second",
        "weekday",
    ]

    feature_names: list[str] = None

    @property
    def remove_digits_transformer(self) -> FunctionTransformer:
        def remove_digits(text: str | ndarray) -> str:
            if isinstance(text, np.ndarray):
                text = str(text[0])
            return re.sub(r"\d+", "", text)

        return FunctionTransformer(
            lambda x: [remove_digits(text) for text in x], validate=False
        )

    @property
    def tfidf_transformer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            strip_accents="ascii",
            lowercase=True,
        )

    @property
    def alias_encoder_transformer(self) -> OneHotEncoder:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    def fit(self, X: list[dict], _: list | None = None) -> "FeatureExtractor":  # noqa: N803
        """Fit the feature extractor on a list of bunq payments.

        - Fit the TFIDF encoder on the description column.
        - Store the feature names

        """
        X = [BunqPayment(**x) for x in X]  # noqa: N806
        # Fit TFIDF encoder on the description column
        descriptions = [x.description for x in X]
        description_encoder = Pipeline(
            [
                (
                    "remove_digits",
                    self.remove_digits_transformer,
                ),
                (
                    "tfidf",
                    self.tfidf_transformer,
                ),
            ]
        )
        description_encoder.fit(descriptions)
        self.description_encoder = description_encoder

        aliasses = [x.alias["display_name"] for x in X] + [
            x.counterparty_alias["display_name"] for x in X
        ]
        alias_encoder = Pipeline(
            [
                (
                    "remove_digits",
                    self.remove_digits_transformer,
                ),
                (
                    "reshape",
                    ReshapeTransformer(),
                ),
                (
                    "onehot",
                    self.alias_encoder_transformer,
                ),
            ]
        )
        alias_encoder.fit(np.array(aliasses))

        self.alias_encoder = alias_encoder

        self.feature_names = [
            *self.COLUMNS,
            *description_encoder[-1].get_feature_names_out(),
            *alias_encoder[-1].get_feature_names_out(),
        ]
        return self

    def transform(self, X: list[BunqPayment]) -> ndarray:  # noqa: N803
        """Transform a list of bunq payments to a numpy array.

        - Add self.COLUMNS as array.
        - Transform the description column with the TFIDF encoder.
        """
        X = [BunqPayment(**x) for x in X]  # noqa: N806
        data = np.array(
            [
                [
                    x.description,
                    x.alias["display_name"],
                    x.counterparty_alias["display_name"],
                    float(x.amount["value"]),
                    int(x.created.hour),
                    int(x.created.minute),
                    int(x.created.second),
                    int(x.created.weekday()),
                ]
                for x in X
            ]
        )
        descriptions = self.description_encoder.transform(data[:, 0]).toarray()
        aliasses = self.alias_encoder.transform(data[:, 1].reshape(-1, 1))
        counterparty_aliasses = self.alias_encoder.transform(data[:, 2].reshape(-1, 1))
        data = np.array(data[:, 3:], dtype=np.float64)

        # Convert the data to a numpy array
        return np.hstack((data, descriptions, aliasses, counterparty_aliasses))
