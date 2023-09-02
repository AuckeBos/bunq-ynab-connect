from typing import Iterable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder

from bunq_ynab_connect.classification.feature_extractor import FeatureExtractor
from bunq_ynab_connect.models.ynab.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class Classifier(ClassifierMixin):
    model: ClassifierMixin
    feature_extractor: FeatureExtractor
    label_encoder: LabelEncoder
    SCORE_NAME = "cohen_kappa"

    def __init__(self, model: ClassifierMixin, label_encoder: LabelEncoder):
        self.model = model
        self.label_encoder = label_encoder
        self.feature_extractor = FeatureExtractor()

    def fit(self, X: Iterable[BunqPayment], y: Iterable[YnabTransaction]):
        X = self.feature_extractor.fit_transform(X)
        y = [transaction.category_name for transaction in y]
        y = self.label_encoder.transform(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.feature_extractor.transform(X)
        return self.model.predict(X)

    def score(self, X, y):
        X = self.feature_extractor.transform(X)
        y = [transaction.category_name for transaction in y]
        y = self.label_encoder.transform(y)
        y_pred = self.model.predict(X)
        cohens_kappa = cohen_kappa_score(y, y_pred)
        return cohens_kappa

    def get_params(self, deep=True):
        return self.model.get_params(deep)
