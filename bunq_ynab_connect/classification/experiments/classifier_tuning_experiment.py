from logging import LoggerAdapter
from typing import List

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from kink import inject
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import mlflow
from bunq_ynab_connect.classification.classifier import Classifier
from bunq_ynab_connect.classification.experiments.base_payment_classification_experiment import (
    BasePaymentClassificationExperiment,
)
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab.matched_transaction import MatchedTransaction
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class ClassifierTuningExperiment(BasePaymentClassificationExperiment):
    """
    Tun one classifier. Ran for the best classifier out of the ClassifierSelectionExperiment.
    """

    HYPERPARAMETER_SPACES = {
        DecisionTreeClassifier().__class__.__name__: {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [3, 5, 10, 20, 50, None],
        },
        RandomForestClassifier().__class__.__name__: {
            "n_estimators": [100, 1000, 2500],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [5, 10, 20, 50, 250, None],
        },
        GradientBoostingClassifier().__class__.__name__: {
            "learning_rate": [0.01, 0.1, 0.5],
            "n_estimators": [100, 1000, 2500],
            "min_samples_split": [2, 5, 10],
        },
        GaussianNB().__class__.__name__: {},
        MLPClassifier().__class__.__name__: {
            "max_iter": [1000],
            "activation": ["tanh", "relu"],
            "solver": ["lbfgs", "sgd"],
            "alpha": [0.01, 0.1, 1],
            "learning_rate": ["contant", "adaptive"],
            "learning_rate_init": [0.01, 0.001, 0.0001],
        },
        ExplainableBoostingClassifier().__class__.__name__: {
            "max_bins": [128, 256, 512],
            "outer_bags": [4, 8, 16],
            "inner_bags": [0, 4, 8],
            "learning_rate": [0.01, 0.1, 0.5],
        },
        # todo: remove
        DecisionTreeClassifier().__class__.__name__: {
            "max_depth": [2, 3],
        },
    }

    clf: BaseEstimator

    @inject
    def __init__(
        self,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        budget_id: str,
        clf: BaseEstimator,
    ):
        super().__init__(storage, logger, budget_id)
        self.clf = clf

    def _run(self, transactions: List[MatchedTransaction]):
        """
        Run the experiment.
        """
        space = self.HYPERPARAMETER_SPACES[self.clf.__class__.__name__]
        classifier = Classifier(self.clf, label_encoder=self.label_encoder)
        grid_search = GridSearchCV(classifier, space)

        train_transactions, test_transactions = train_test_split(
            transactions, test_size=0.2, random_state=42
        )
        X_train = [t.bunq_payment for t in train_transactions]
        X_test = [t.bunq_payment for t in test_transactions]
        y_train = [t.ynab_transaction for t in train_transactions]
        y_test = [t.ynab_transaction for t in test_transactions]
        self.log_transactions(train_transactions, "train_ids")
        self.log_transactions(test_transactions, "test_ids")

        grid_search.fit(X_train, y_train)
        best_score = grid_search.best_score_
        mlflow.log_metric(Classifier.SCORE_NAME, best_score)

        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)
        score = self.score(y_test, y_pred)
        print(f"Score of best clf: {score}")
        mlflow.log_metric("cohens_kappa", score)
        self.select_best_run()
        self.grid_search = grid_search
