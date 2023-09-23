from typing import List

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from bunq_ynab_connect.models.bunq_payment import BunqPayment


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract features from the bunq payments.


    """

    encoder: TfidfVectorizer

    COLUMNS = [
        "description",
        "amount",
        "hour",
        "minute",
        "second",
        "weekday",
    ]

    feature_names: List[str] = None

    def fit(self, X: List[dict], y=None) -> "FeatureExtractor":
        """
        Fit the feature extractor on a list of bunq payments.
        - Fit the TFIDF encoder on the description column.
        - Store the feature names

        """
        X = [BunqPayment(**x) for x in X]
        # Fit TFIDF encoder on the description column
        descriptions = [x.description for x in X]
        encoder = TfidfVectorizer(
            strip_accents="ascii",
            lowercase=True,
        )
        encoder.fit(descriptions)
        self.feature_names = [*self.COLUMNS, *encoder.get_feature_names_out()]
        self.encoder = encoder
        return self

    def transform(self, X: List[BunqPayment]) -> ndarray:
        """
        Transform a list of bunq payments to a numpy array.
        - Add self.COLUMNS as array.
        - Transform the description column with the TFIDF encoder.
        """
        X = [BunqPayment(**x) for x in X]
        data = np.array(
            [
                [
                    x.description,
                    float(x.amount["value"]),
                    int(x.created.hour),
                    int(x.created.minute),
                    int(x.created.second),
                    int(x.created.weekday()),
                ]
                for x in X
            ]
        )
        descriptions = self.encoder.transform(data[:, 0]).toarray()
        data = np.array(data[:, 1:], dtype=np.float64)

        # Convert the data to a numpy array
        data = np.hstack((data, descriptions))
        return data
