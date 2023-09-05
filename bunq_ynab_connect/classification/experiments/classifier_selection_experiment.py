from typing import List

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
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
from bunq_ynab_connect.models.ynab.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab.matched_transaction import MatchedTransaction
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class ClassifierSelectionExperiment(BasePaymentClassificationExperiment):
    """
    Select the best classifier for the job. Use default parameters.
    For each classifier, train and test the model. Use K-fold cross validation.
    """

    N_FOLDS = 3
    RANDOM_STATE = 42

    CLASSIFIERS = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        MLPClassifier(max_iter=1000),
        ExplainableBoostingClassifier(),
    ]

    def _run(self, transactions: List[MatchedTransaction]):
        """
        Run the experiment.
        """
        results = []
        for classifier in self.CLASSIFIERS:
            with mlflow.start_run(
                run_name=classifier.__class__.__name__, nested=True
            ) as run:
                self.run_classifier(classifier, transactions)
        return results

    def run_classifier(
        self, model: ClassifierMixin, transactions: List[MatchedTransaction]
    ):
        """
        Run the experiment for a single classifier.
        """
        X = np.array([t.bunq_payment for t in transactions])
        y = np.array([t.ynab_transaction for t in transactions])
        y = self.label_encoder.fit_transform(y)
        classifier = self.create_pipeline(model)
        k_fold = StratifiedKFold(
            n_splits=self.N_FOLDS, shuffle=True, random_state=self.RANDOM_STATE
        )
        scores = cross_val_score(classifier, X, y, cv=k_fold, n_jobs=-1)
        mlflow.log_text(str(scores), "scores.txt")
        avg_score = np.mean(scores)
        mlflow.log_metric(Classifier.SCORE_NAME, avg_score)
        return avg_score
