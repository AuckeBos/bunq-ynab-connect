from typing import List

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
    label_encoder: OneHotEncoder

    def __init__(self, model: ClassifierMixin, **kwargs):
        self.model = model
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = OneHotEncoder(handle_unknown="ignore")

    def fit(self, X: List[BunqPayment], y: List[YnabTransaction]):
        self.model.fit(X, y)
        self.feature_extractor.fit(X)
        self.label_encoder.fit([transaction.category_name for transaction in y])
        return self

    def predict(self, X):
        X = self.feature_extractor.transform(X)
        return self.model.predict(X)

    def score(self, X, y):
        X = self.feature_extractor.transform(X)
        y_pred = self.model.predict(X)
        cohens_kappa = cohen_kappa_score(y, y_pred)
        return cohens_kappa

    def get_params(self, deep=True):
        return self.model.get_params(deep)
