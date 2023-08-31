from typing import List

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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
from bunq_ynab_connect.models.ynab.ynab_transaction import YnabTransaction


class ClassifierSelectionExperiment(BasePaymentClassificationExperiment):
    """
    Select the best classifier for the job. Use default parameters.
    For each classifier, train and test the model. Use K-fold cross validation.
    """

    N_FOLDS = 5

    CLASSIFIERS = [
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC(),
        GaussianNB(),
        DecisionTreeClassifier(),
        MLPClassifier(max_iter=1000),
    ]

    def _run(self, X: List[BunqPayment], y: List[YnabTransaction]):
        """
        Run the experiment.
        """
        results = []
        for classifier in self.CLASSIFIERS:
            with mlflow.start_run(
                run_name=classifier.__class__.__name__, nested=True
            ) as run:
                results.append(self.run_classifier(classifier, X, y))
        return results

    def run_classifier(
        self, model: ClassifierMixin, X: List[BunqPayment], y: List[YnabTransaction]
    ):
        """
        Run the experiment for a single classifier.
        """
        classifier = Classifier(model)
        cross_val_score(classifier, X, y, cv=self.N_FOLDS)
