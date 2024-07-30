from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from interpret.glassbox import ExplainableBoostingClassifier
from kink import inject
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import mlflow
from bunq_ynab_connect.classification.experiments.base_payment_classification_experiment import (  # noqa: E501
    BasePaymentClassificationExperiment,
)

if TYPE_CHECKING:
    from logging import LoggerAdapter

    import numpy as np

    from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


class ClassifierTuningExperiment(BasePaymentClassificationExperiment):
    """Tune one classifier. Ran for the classifier of  ClassifierSelectionExperiment.

    Attributes
    ----------
        N_FOLDS: Amount of folds to use for K-fold cross validation
        HYPERPARAMETER_SPACES: Dictionary containing the hyperparameter spaces for each
            classifier
        grid_search: GridSearchCV object used to find the best parameters
        clf: Classifier to tune

    """

    N_FOLDS = 3

    HYPERPARAMETER_SPACES: ClassVar[dict[str, dict[str, Any]]] = {
        DecisionTreeClassifier().__class__.__name__: {
            "max_depth": [3, 5, 10, 20, 50, None],
        },
        RandomForestClassifier().__class__.__name__: {
            "n_estimators": [100, 1000, 2500],
            "max_depth": [5, 10, 20, 50],
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
            "solver": ["sgd"],
            "alpha": [0.01, 0.1, 1],
            "learning_rate": ["adaptive"],
            "learning_rate_init": [0.001],
        },
        ExplainableBoostingClassifier().__class__.__name__: {
            "max_bins": [128, 256, 512],
            "learning_rate": [0.01, 0.1, 0.5],
        },
    }

    clf: Any
    grid_search: GridSearchCV

    @inject
    def __init__(
        self,
        budget_id: str,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        *,
        clf: Any,  # noqa: ANN401
    ):
        super().__init__(budget_id, storage, logger)
        self.grid_search = None
        self.clf = clf

    def _run(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        # Prefix the hyperparameter space with the classifier name.
        space = {
            f"classifier__{key}": value
            for key, value in self.HYPERPARAMETER_SPACES[self.clf.__name__].items()
        }
        classifier = self.clf()
        pipeline = self.create_pipeline(classifier)
        self.grid_search = GridSearchCV(
            pipeline,
            space,
            cv=self.N_FOLDS,
            scoring=make_scorer(cohen_kappa_score),
            n_jobs=-1,
        )
        self.grid_search.fit(X, y)
        score = self.grid_search.best_score_
        self.logger.info("Score of best clf: %s", score)
        mlflow.log_metric("cohens_kappa", score)

    def get_best_parameters(self) -> dict:
        """After running the experiment, get the best parameters."""
        if not self.grid_search:
            return None
        best_params = self.grid_search.best_params_
        # Remove prefix from the best parameters.
        best_params = {
            key.replace("classifier__", ""): value for key, value in best_params.items()
        }
        best_score = self.grid_search.best_score_
        self.logger.info(
            "The best parameters are: %s, with a score of %s", best_params, best_score
        )
        return best_params
