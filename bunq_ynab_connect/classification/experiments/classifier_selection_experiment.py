import multiprocessing
from logging import LoggerAdapter

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from kink import inject
from mlflow.client import MlflowClient
from prefect import task
from sklearn.base import ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
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
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.matched_transaction import MatchedTransaction
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction


class ClassifierSelectionExperiment(BasePaymentClassificationExperiment):
    """
    Select the best classifier for the job. Use default parameters.
    For each classifier, train and test the model. Use K-fold cross validation.

    Attributes:
        N_FOLDS: Amount of folds to use for K-fold cross validation
        RANDOM_STATE: Random state to use for K-fold cross validation
        run_id: ID of the current run. Set upon run()
        threads: Not used
        CLASSIFIERS: List of classifiers to test
    """

    N_FOLDS = 3
    RANDOM_STATE = 42
    run_id: str
    threads: int

    CLASSIFIERS = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        MLPClassifier(max_iter=1000),
        ExplainableBoostingClassifier(),
    ]

    @inject
    def __init__(
        self,
        budget_id: str,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        threads: int = None,
    ):
        super().__init__(budget_id, storage, logger)
        self.parent_run_id = None
        if not threads:
            threads = multiprocessing.cpu_count() - 1
        self.threads = threads

    def _run(self, X: np.ndarray, y: np.ndarray):
        """
        Run the experiment.
        """

        for classifier in self.CLASSIFIERS:
            with mlflow.start_run(run_name=classifier.__class__.__name__, nested=True):
                self.run_classifier(classifier, X, y)

    def run_classifier(self, model: ClassifierMixin, X: np.ndarray, y: np.ndarray):
        """
        Run the experiment for a single classifier.
        - Create the pipeline
        - Use Kfold
        - Score and log the mean score

        Use the client to log, to prevent issues due to threading.
        """
        mlflow.set_tag("classifier", model.__class__.__name__)
        classifier = self.create_pipeline(model)
        k_fold = StratifiedKFold(
            n_splits=self.N_FOLDS, shuffle=True, random_state=self.RANDOM_STATE
        )
        scores = cross_val_score(
            classifier,
            X,
            y,
            cv=k_fold,
            n_jobs=-1,
            scoring=make_scorer(cohen_kappa_score),
        )
        mlflow.log_text(str(scores), "scores.txt")
        avg_score = np.mean(scores)
        mlflow.log_metric("cohen_kappa", avg_score)
        mlflow.sklearn.log_model(classifier, "model")
        return avg_score

    def get_best_classifier(self) -> ClassifierMixin:
        """
        Select the best classifier based on the score.
        """
        if not self.parent_run_id:
            return None
        client = MlflowClient()
        experiment_id = client.get_experiment_by_name(
            self.get_experiment_name()
        ).experiment_id
        runs = client.search_runs(
            experiment_id,
            filter_string=f"tags.mlflow.parentRunId = '{self.parent_run_id}'",
            order_by=["metrics.cohens_kappa DESC"],
        )
        best_run = runs[0]
        best_classifier_class_name = best_run.data.tags["classifier"]
        score = best_run.data.metrics["cohen_kappa"]
        self.logger.info(
            f"The best classifier is: {best_classifier_class_name}, with a score of {score}"
        )
        best_classifier_class = eval(best_classifier_class_name)
        return best_classifier_class
