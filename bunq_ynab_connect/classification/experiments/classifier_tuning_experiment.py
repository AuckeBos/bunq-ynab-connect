from logging import LoggerAdapter

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


class ClassifierTuningExperiment(BasePaymentClassificationExperiment):
    """
    Tun one classifier. Ran for the best classifier out of the ClassifierSelectionExperiment.
    """

    N_FOLDS = 3

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
    }

    clf: type[ClassifierMixin]

    @inject
    def __init__(
        self,
        budget_id: str,
        clf: type[ClassifierMixin],
        storage: AbstractStorage,
        logger: LoggerAdapter,
    ):
        super().__init__(budget_id, storage, logger)
        self.clf = clf

    def _run(self, X: np.ndarray, y: np.ndarray):
        """
        Run the experiment.
        """
        space = {
            f"classifier__{key}": value
            for key, value in self.HYPERPARAMETER_SPACES[self.clf.__name__].items()
        }
        classifier = self.clf()
        pipeline = self.create_pipeline(classifier)
        grid_search = GridSearchCV(
            pipeline, space, cv=self.N_FOLDS, scoring=make_scorer(cohen_kappa_score)
        )
        grid_search.fit(X, y)
        grid_search.best_estimator_
        score = grid_search.best_score_
        print(f"Score of best clf: {score}")
        mlflow.log_metric("cohens_kappa", score)
