from logging import LoggerAdapter
from typing import Any

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from kink import inject
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import mlflow
from bunq_ynab_connect.classification.budget_category_encoder import (
    BudgetCategoryEncoder,
)
from bunq_ynab_connect.classification.classifier import Classifier
from bunq_ynab_connect.classification.experiments.base_payment_classification_experiment import (
    BasePaymentClassificationExperiment,
)
from bunq_ynab_connect.classification.feature_extractor import FeatureExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab.matched_transaction import MatchedTransaction
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class FullTrainingExperiment(BasePaymentClassificationExperiment):
    """
    An experiment to run a classifier with the best parameters on all data.
    The result of this experiment is the final model, which can be deployed.
    """

    model: ClassifierMixin

    @inject
    def __init__(
        self,
        budget_id: str,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        *,
        clf: Any,
        parameters: dict
    ):
        super().__init__(budget_id, storage, logger)
        self.model = clf(**parameters)

    def _run(self, X: np.ndarray, y: np.ndarray):
        classifier = self.create_pipeline(self.model)
        classifier.fit(X, y)
        mlflow.sklearn.log_model(classifier, "model")
        self.logger.info("Finished training")
