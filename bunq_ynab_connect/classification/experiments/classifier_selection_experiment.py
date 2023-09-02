from typing import List

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
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

    N_FOLDS = 3
    RANDOM_STATE = 42

    CLASSIFIERS = [
        KNeighborsClassifier(n_neighbors=3),
        DecisionTreeClassifier(),
        LogisticRegression(max_iter=1000),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        SVC(),
        GaussianNB(),
        MLPClassifier(max_iter=1000),
        ExplainableBoostingClassifier(),
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
                self.run_classifier(classifier, X, y)
        return results

    def run_classifier(
        self, model: ClassifierMixin, X: List[BunqPayment], y: List[YnabTransaction]
    ):
        """
        Run the experiment for a single classifier.
        """
        X, y = np.array(X), np.array(y)
        label_encoder = LabelEncoder()
        label_encoder.fit([transaction.category_name for transaction in y])

        scores = []
        classifier = Classifier(model, label_encoder=label_encoder)
        k_fold = KFold(
            n_splits=self.N_FOLDS, shuffle=True, random_state=self.RANDOM_STATE
        )
        for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            classifier.fit(X_train, y_train)
            scores.append(classifier.score(X_test, y_test))
            self.logger.info(
                f"Fold {i + 1} for {model.__class__.__name__}: {scores[-1]}"
            )
        mlflow.log_text(str(scores), "scores.txt")
        avg_score = np.mean(scores)
        mlflow.log_metric(Classifier.SCORE_NAME, avg_score)
        return avg_score
