import multiprocessing
from multiprocessing.pool import ThreadPool

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from mlflow.client import MlflowClient
from prefect import task
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

    def _run(self, X: np.ndarray, y: np.ndarray):
        """
        Run the experiment. Use a Thread Pool to run the classifiers in parallel.
        The amount of threads is the amount of CPU cores minus 1.
        """

        def run_for_one_classifier(
            self, classifier: ClassifierMixin, X: np.ndarray, y: np.ndarray
        ):
            """
            Run the experiment for a single classifier.
            Use the client instead of mlflow, to prevent threading issues and non-closed runs.
            """
            client = MlflowClient()
            experiment_id = client.get_experiment_by_name(
                self.get_experiment_name()
            ).experiment_id
            run = client.create_run(
                experiment_id=experiment_id,
                tags={"mlflow.parentRunId": self.run_id},
                run_name=classifier.__class__.__name__,
            )
            try:
                self.run_classifier(classifier, X, y)
            finally:
                client.set_terminated(run.info.run_id)

        pool = ThreadPool(processes=multiprocessing.cpu_count() - 1)
        pool.map(
            lambda classifier: run_for_one_classifier(self, classifier, X, y),
            self.CLASSIFIERS,
        )

    def run_classifier(self, model: ClassifierMixin, X: np.ndarray, y: np.ndarray):
        """
        Run the experiment for a single classifier.
        - Create the pipeline
        - Use Kfold
        - Score and log the mean score
        """
        classifier = self.create_pipeline(model)
        k_fold = StratifiedKFold(
            n_splits=self.N_FOLDS, shuffle=True, random_state=self.RANDOM_STATE
        )
        scores = cross_val_score(classifier, X, y, cv=k_fold, n_jobs=-1)
        mlflow.log_text(str(scores), "scores.txt")
        avg_score = np.mean(scores)
        mlflow.log_metric(Classifier.SCORE_NAME, avg_score)
        return avg_score
