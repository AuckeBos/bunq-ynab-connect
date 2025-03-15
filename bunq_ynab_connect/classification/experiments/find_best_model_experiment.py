import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from imblearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)
from sklearn.pipeline import FeatureUnion

import mlflow
from bunq_ynab_connect.classification.experiments.base_payment_classification_experiment import (  # noqa: E501
    BasePaymentClassificationExperiment,
)
from bunq_ynab_connect.classification.preprocessing.alias_features import AliasFeatures
from bunq_ynab_connect.classification.preprocessing.counterparty_similarity_features import (
    CounterpartySimilarityFeatures,
)
from bunq_ynab_connect.classification.preprocessing.description_features import (
    DescriptionFeatures,
)
from bunq_ynab_connect.classification.preprocessing.dict_to_transaction_transformer import (
    DictToTransactionTransformer,
)
from bunq_ynab_connect.classification.preprocessing.over_sampler import OverSampler
from bunq_ynab_connect.classification.preprocessing.simple_features import (
    SimpleFeatures,
)
from bunq_ynab_connect.classification.preprocessing.under_sampler import UnderSampler


class FindBestModelExperiment(BasePaymentClassificationExperiment):
    """Experiment to find and train the best model, using hyperopt.

    Use hyperopt to find the best score over a search space. Search space
    contains different models and different parameters. It also includes
    parameters to configure the feature extraction and sampling.
    """

    def _train_and_score(
        self, pipeline: Pipeline, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Train the pipeline on the data and return the f1 score.

        Use stratified 90-10 split. Log the metrics in mlflow.
        Return format as expected by hyperopt. Use macro f1 score as the
        objective to optimize.
        """
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
        train_index, test_index = next(splitter.split(X, y))
        X_train, X_test = X[train_index], X[test_index]  # noqa: N806
        y_train, y_test = y[train_index], y[test_index]
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_pred = pipeline.predict(X_test)
        mlflow.log_metrics(
            {
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
                "test_f1_micro": f1_score(y_test, y_pred, average="micro"),
                "test_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_f1_macro": f1_score(y_train, y_train_pred, average="macro"),
                "train_f1_micro": f1_score(y_train, y_train_pred, average="micro"),
                "train_f1_weighted": f1_score(
                    y_train, y_train_pred, average="weighted"
                ),
                "train_balanced_accuracy": balanced_accuracy_score(
                    y_train, y_train_pred
                ),
            }
        )
        return -f1_score(y_test, y_pred, average="macro")

    def _find_best_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Search space with hyperopt.

        Finds the best model and its best configuration.
        """
        # Autolog doesnt work well with hyperopt
        mlflow.sklearn.autolog(disable=True)

        def objective(params: dict) -> dict:
            """The objective function to optimize.

            Start a child run. Create a pipeline based on the parameters
            and train it. Log the parameters and the result in mlflow.
            """  # noqa: D401
            with mlflow.start_run(nested=True):
                classifier: ClassifierMixin = eval(params["classifier"])()  # noqa: S307
                pipeline = self.create_pipeline(classifier)
                pipeline.set_params(**params["parameters"])
                mlflow.log_params(pipeline.get_params())
                result = self._train_and_score(pipeline, X, y)

                return {
                    "loss": result,
                    "status": STATUS_OK,
                }

        classifier_options = [RandomForestClassifier.__name__]
        space = hp.choice(
            "classifier",
            [
                # One item for each model type
                {
                    "classifier": RandomForestClassifier.__name__,
                    "parameters": {
                        **self._feature_extraction_search_space,
                        **self._sampling_search_space,
                        "classifier__max_depth": hp.uniformint("max_depth", 50, 500),
                        "classifier__min_samples_split": hp.uniformint(
                            "min_samples_split", 2, 15
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
            max_evals=500,
            trials=trials,
        )
        best_config["classifier"] = classifier_options[best_config["classifier"]]
        mlflow.log_params(best_config)

    @property
    def _feature_extraction_search_space(self) -> dict:
        """The search space configuration for the feature extraction."""
        return {
            "feature_extractor__description_features__max_features": scope.int(
                hp.uniform(
                    "max_features",
                    750,
                    3000,
                )
            ),
            "feature_extractor__description_features__enabled": hp.choice(
                "descriptions_enabled",
                [True],
            ),
            "feature_extractor__alias_features__top_categories": hp.uniformint(
                "alias_top_categories", 20, 80
            ),
            "feature_extractor__alias_features__enabled": hp.choice(
                "alias_enabled",
                [False],
            ),
            "feature_extractor__counterparty_similarity_features__top_categories": hp.uniformint(
                "counterparty_top_categories", 20, 80
            ),
            "feature_extractor__counterparty_similarity_features__enabled": hp.choice(
                "counterparty_enabled",
                [True, False],
            ),
        }

    @property
    def _sampling_search_space(self) -> dict:
        """The search space configuration for the sampling."""
        return {
            "under_sampler__percentile": hp.uniform("under_percentile", 0.9, 1.0),
            "over_sampler__k_neighbours": scope.int(hp.uniform("k_neighbours", 0, 5)),
            "over_sampler__percentile": hp.uniform("over_percentile", 0.2, 0.8),
        }

    def create_pipeline(self, classifier: ClassifierMixin) -> Pipeline:
        """Create the pipeline with the given classifier.

        - Convert the input to a list of transactions
        - Extract all features, and concatenate them horizontally
        - Undersample the majority class
        - Oversample the minority class
        - Train the classifier
        """
        return Pipeline(
            [
                ("to_transaction", DictToTransactionTransformer()),
                (
                    "feature_extractor",
                    FeatureUnion(
                        transformer_list=[
                            ("simple_features", SimpleFeatures()),
                            ("description_features", DescriptionFeatures()),
                            ("alias_features", AliasFeatures()),
                            (
                                "counterparty_similarity_features",
                                CounterpartySimilarityFeatures(),
                            ),
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

    def _run(self, X: np.ndarray, y: np.ndarray) -> None:
        X, y = self._remove_singleton_categories(X, y)  # noqa: N806
        self._find_best_model(X, y)

    def _remove_singleton_categories(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove all categories that only have one item.

        We can not split stratified if a class only occurs once.
        """
        unique, counts = np.unique(y, return_counts=True)
        X = X[np.isin(y, unique[counts > 1])]  # noqa: N806
        y = y[np.isin(y, unique[counts > 1])]
        return X, y
