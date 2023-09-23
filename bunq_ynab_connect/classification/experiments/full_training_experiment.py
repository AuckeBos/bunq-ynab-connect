from logging import LoggerAdapter
from typing import Any

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from kink import inject
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
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
from bunq_ynab_connect.classification.deployable_mlflow_model import (
    DeployableMlflowModel,
)
from bunq_ynab_connect.classification.experiments.base_payment_classification_experiment import (
    BasePaymentClassificationExperiment,
)
from bunq_ynab_connect.classification.feature_extractor import FeatureExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.general import object_to_mlflow
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.matched_transaction import MatchedTransaction
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction


class FullTrainingExperiment(BasePaymentClassificationExperiment):
    """
    An experiment to run a classifier with the best parameters on all data.
    The result of this experiment is the final model, which can be deployed.
    """

    model: ClassifierMixin
    RANDOM_STATE = 42

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
        """
        Split the data once. Train the classifier on the training data and evaluate on the test data.
        Log the model as a DeployableMlflowModel.
        """
        classifier = self.create_pipeline(self.model)

        train_idx, test_idx = next(
            StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.RANDOM_STATE
            ).split(X, y)
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        score = cohen_kappa_score(y_test, y_pred)
        mlflow.log_metric("cohen_kappa", score)
        mlflow.sklearn.log_model(classifier, "model")
        object_to_mlflow(self.label_encoder, "label_encoder")

        artifact_uri = mlflow.active_run().info.artifact_uri
        artifacts = {
            "model_path": artifact_uri + "/model",
            "label_encoder_path": artifact_uri + "/label_encoder",
        }
        input_example = [X_train[0]]
        mlflow.pyfunc.log_model(
            artifact_path="deployable_model",
            python_model=DeployableMlflowModel(),
            artifacts=artifacts,
            input_example=input_example,
            registered_model_name=self.budget_id,
        )

        self.logger.info("Finished training")
