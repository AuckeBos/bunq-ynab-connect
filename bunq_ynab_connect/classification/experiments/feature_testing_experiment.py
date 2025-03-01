from typing import ClassVar

import numpy as np
from hyperopt.pyll.base import scope
from sklearn.pipeline import FeatureUnion

from bunq_ynab_connect.classification.preprocessing.over_sampler import OverSampler
from bunq_ynab_connect.classification.preprocessing.under_sampler import UnderSampler
from bunq_ynab_connect.models.bunq_payment import BunqPayment

# prevent remove imports when commented out
if True:
    from imblearn.over_sampling import SMOTE, RandomOverSampler  # noqa: F401
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

import mlflow
from bunq_ynab_connect.classification.experiments.base_payment_classification_experiment import (  # noqa: E501
    BasePaymentClassificationExperiment,
)
from bunq_ynab_connect.classification.preprocessing.alias_features import AliasFeatures
from bunq_ynab_connect.classification.preprocessing.description_features import (
    DescriptionFeatures,
)
from bunq_ynab_connect.classification.preprocessing.simple_features import (
    SimpleFeatures,
)


class ToTransactionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        return [BunqPayment(**x) for x in X]


class FeatureTestingExperiment(BasePaymentClassificationExperiment):
    """todo: write doc"""

    N_FOLDS = 3
    RANDOM_STATE = 42
    CLASSIFIERS: ClassVar[list[ClassifierMixin]] = [
        # DecisionTreeClassifier(),
        # RandomForestClassifier(),
        GaussianNB(),
        MLPClassifier(max_iter=1000, alpha=0.01),
        DecisionTreeClassifier(
            class_weight="balanced", max_depth=50, max_features="sqrt"
        ),
        RandomForestClassifier(
            class_weight="balanced", max_depth=50, max_features="sqrt"
        ),
    ]

    def tune_single_classifier(self, X: np.ndarray, y: np.ndarray) -> None:
        def objective(params: dict) -> dict:
            with mlflow.start_run(nested=True):
                classifier: ClassifierMixin = eval(params["classifier"])()
                pipeline = self.create_pipeline(classifier)
                pipeline.set_params(**params["parameters"])
                mlflow.log_params(pipeline.get_params())
                k_fold = StratifiedKFold(
                    n_splits=self.N_FOLDS, shuffle=True, random_state=self.RANDOM_STATE
                )
                scores = cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=k_fold,
                    n_jobs=-1,
                    scoring={
                        "f1_macro": make_scorer(f1_score, average="macro"),
                        "f1_micro": make_scorer(f1_score, average="micro"),
                        "f1_weighted": make_scorer(f1_score, average="weighted"),
                        "accuracy": make_scorer(accuracy_score),
                        "balanced_accuracy": make_scorer(balanced_accuracy_score),
                    },
                    return_train_score=True,
                )
                for type_ in ["train", "test"]:
                    for metric in [
                        "accuracy",
                        "f1_macro",
                        "f1_micro",
                        "f1_weighted",
                        "balanced_accuracy",
                    ]:
                        mlflow.log_metric(
                            f"{type_}_{metric}", np.mean(scores[f"{type_}_{metric}"])
                        )
                return {
                    "loss": -np.mean(scores["test_f1_macro"]),
                    "status": STATUS_OK,
                }

        classifier_options = [RandomForestClassifier.__name__]
        space = hp.choice(
            "classifier",
            [
                {
                    "classifier": RandomForestClassifier.__name__,
                    "parameters": {
                        "feature_extractor__description_features__max_features": scope.int(
                            hp.uniform(
                                "max_features",
                                100,
                                2000,
                            )
                        ),
                        "classifier__max_depth": scope.int(
                            hp.uniform("max_depth", 5, 500)
                        ),
                        "classifier__min_samples_split": scope.int(
                            hp.uniform("min_samples_split", 2, 100)
                        ),
                        "under_sampler__percentile": hp.uniform(
                            "under_percentile", 0.0, 1.0
                        ),
                        "over_sampler__k_neighbours": scope.int(
                            hp.uniform("k_neighbours", 0, 5)
                        ),
                        "over_sampler__percentile": hp.uniform(
                            "over_percentile", 0.0, 1.0
                        ),
                    },
                },
            ],
        )

        trials = Trials()
        best_config = fmin(
            objective,
            space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
        )
        best_config["classifier"] = classifier_options[best_config["classifier"]]
        mlflow.log_params(best_config)

    def create_pipeline(self, classifier: ClassifierMixin) -> Pipeline:
        return Pipeline(
            [
                ("to_transaction", ToTransactionTransformer()),
                (
                    "feature_extractor",
                    FeatureUnion(
                        transformer_list=[
                            ("simple_features", SimpleFeatures()),
                            ("description_features", DescriptionFeatures()),
                            ("alias_features", AliasFeatures()),
                        ]
                    ),
                ),
                (
                    "under_sampler",
                    UnderSampler(),
                ),
                ("over_sampler", OverSampler()),
                ("classifier", classifier),
            ]
        )

    def _run(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
        self.tune_single_classifier(X, y)
