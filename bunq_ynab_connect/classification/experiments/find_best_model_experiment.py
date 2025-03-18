import contextlib
from logging import LoggerAdapter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from imblearn.pipeline import Pipeline
from kink import inject
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)

import mlflow
from bunq_ynab_connect.classification.experiments.base_payment_classification_experiment import (  # noqa: E501
    BasePaymentClassificationExperiment,
)
from bunq_ynab_connect.helpers.general import object_to_mlflow

if TYPE_CHECKING:
    from sklearn.base import ClassifierMixin


@contextlib.contextmanager
def progress_callback(initial: int, total: int):  # noqa: ANN201
    """Progress callback for hyperopt.

    Logs the progress of the hyperopt optimization to the default application logger.
    """

    class ProgressContext:
        n: int
        logger: LoggerAdapter

        @inject
        def __init__(self, logger: LoggerAdapter):
            self.n = initial
            self.logger = logger

        def update(self, n: int) -> None:
            self.n += n
            self.logger.info("Step %d/%d", self.n, total)

    yield ProgressContext()


class FindBestModelExperiment(BasePaymentClassificationExperiment):
    """Experiment to find and train the best model, using hyperopt.

    Use hyperopt to find the best score over a search space. Search space
    contains different models and different parameters. It also includes
    parameters to configure the feature extraction and sampling.

    Attributes
    ----------
        max_runs: Maximum number of evaluations to run.

    """

    max_runs: int = 250

    def _run(self, X: np.ndarray, y: np.ndarray) -> None:
        best_config = self._find_best_model(*self._remove_singleton_categories(X, y))
        self._train_and_log_best_model(X, y, best_config)

    def _find_best_model(self, X: np.ndarray, y: np.ndarray) -> Trials:
        """Search space with hyperopt.

        Finds the best model and its best configuration.

        Returns the best config.
        """
        # Autolog doesnt work well with hyperopt
        mlflow.sklearn.autolog(disable=True)

        def objective(params: dict) -> dict:
            """The objective function to optimize.

            Start a child run. Create a pipeline based on the parameters
            and train it. Log the parameters and the result in mlflow.
            """  # noqa: D401
            with mlflow.start_run(nested=True) as run:
                classifier: ClassifierMixin = eval(params["classifier"])()  # noqa: S307
                pipeline = self.create_pipeline(classifier)
                pipeline.set_params(**params["parameters"])
                mlflow.log_params(pipeline.get_params())
                result = self._train_and_score(pipeline, X, y)

                return {
                    "loss": result,
                    "status": STATUS_OK,
                    "parameters": params["parameters"],
                    "classifier": params["classifier"],
                    "run_id": run.info.run_id,
                }

        space = hp.choice(
            "classifier",
            [
                # One item for each model type
                {
                    "classifier": RandomForestClassifier.__name__,
                    "parameters": {
                        **self._feature_extraction_search_space("rf"),
                        **self._sampling_search_space("rf"),
                        "classifier__max_depth": hp.uniformint(
                            "classifier__max_depth", 50, 500
                        ),
                        "classifier__min_samples_split": hp.uniformint(
                            "classifier__min_samples_split", 2, 15
                        ),
                    },
                },
                # {
                #     "classifier": KNeighborsClassifier.__name__,
                #     "parameters": {
                #         **self._feature_extraction_search_space("knn"),
                #         **self._sampling_search_space("knn"),
                #         "classifier__n_neighbors": hp.uniformint(
                #             "classifier__n_neighbors", 1, 20
                #         ),
                #         "classifier__weights": hp.choice(
                #             "classifier__weights", ["uniform", "distance"]
                #         ),
                #         "classifier__algorithm": hp.choice(
                #             "classifier__algorithm", ["auto", "ball_tree", "kd_tree"]
                #         ),
                #     },
                # },
                # {
                #     "classifier": GradientBoostingClassifier.__name__,
                #     "parameters": {
                #         **self._feature_extraction_search_space("gb"),
                #         **self._sampling_search_space("gb"),
                #         "classifier__learning_rate": hp.uniform(
                #             "classifier__learning_rate", 0.01, 0.5
                #         ),
                #         "classifier__n_estimators": hp.uniformint(
                #             "classifier__n_estimators", 50, 500
                #         ),
                #     },
                # },
                # {
                #     "classifier": MLPClassifier.__name__,
                #     "parameters": {
                #         **self._feature_extraction_search_space("mlp"),
                #         **self._sampling_search_space("mlp"),
                #         "classifier__hidden_layer_sizes": hp.uniformint(
                #             "classifier__hidden_layer_sizes", 50, 500
                #         ),
                #         "classifier__activation": hp.choice(
                #             "classifier__activation",
                #             ["identity", "logistic", "tanh", "relu"],
                #         ),
                #         "classifier__alpha": hp.uniform(
                #             "classifier__alpha", 0.0001, 0.1
                #         ),
                #     },
                # },
            ],
        )

        # uri = "mongodb://uname:pw@localhost:27017/hyperopt/jobs"
        # trials = MongoTrials(uri)
        trials = Trials()
        fmin(
            objective,
            space,
            algo=tpe.suggest,
            max_evals=self.max_runs,
            trials=trials,
            show_progressbar=progress_callback,
        )
        return trials

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
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
            "test_f1_micro": f1_score(y_test, y_pred, average="micro"),
            "test_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1_macro": f1_score(y_train, y_train_pred, average="macro"),
            "train_f1_micro": f1_score(y_train, y_train_pred, average="micro"),
            "train_f1_weighted": f1_score(y_train, y_train_pred, average="weighted"),
            "train_balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        }
        # The final_score key is used by the deployer, to decide model improvement
        metrics["final_score"] = metrics["test_f1_macro"]
        mlflow.log_metrics(metrics)
        return -metrics["final_score"]

    def _feature_extraction_search_space(self, classifier_name: str) -> dict:
        """The search space configuration for the feature extraction."""  # noqa: D401
        return {
            "feature_extractor__description_features__max_features": scope.int(
                hp.uniform(
                    f"{classifier_name}_feature_extractor__description_features__max_features",
                    750,
                    3000,
                )
            ),
            "feature_extractor__description_features__enabled": hp.choice(
                f"{classifier_name}_feature_extractor__description_features__enabled",
                [True],
            ),
            "feature_extractor__alias_features__top_categories": hp.uniformint(
                f"{classifier_name}_feature_extractor__alias_features__top_categories",
                20,
                80,
            ),
            "feature_extractor__alias_features__enabled": hp.choice(
                f"{classifier_name}_feature_extractor__alias_features__enabled",
                [False],
            ),
            "feature_extractor__counterparty_similarity_features__top_categories": hp.uniformint(  # noqa: E501
                f"{classifier_name}_feature_extractor__counterparty_similarity_features__top_categories",
                20,
                80,
            ),
            "feature_extractor__counterparty_similarity_features__enabled": hp.choice(
                f"{classifier_name}_feature_extractor__counterparty_similarity_features__enabled",
                [True],
            ),
        }

    def _sampling_search_space(self, classifier_name: str) -> dict:
        """The search space configuration for the sampling."""  # noqa: D401
        return {
            "under_sampler__percentile": hp.uniform(
                f"{classifier_name}_under_sampler__percentile", 0.9, 1.0
            ),
            "over_sampler__k_neighbours": hp.uniformint(
                f"{classifier_name}_over_sampler__k_neighbours", 0, 5
            ),
            "over_sampler__percentile": hp.uniform(
                f"{classifier_name}_over_sampler__percentile", 0.2, 0.8
            ),
        }

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

    def _train_and_log_best_model(
        self, X: np.ndarray, y: np.ndarray, trials: Trials
    ) -> None:
        """Train the best model with the best configuration.

        Log to mlflow.

        - Load classifier and params from best trial
        - Fit and log the model
        - Log parameters
        - Log the same metris of the best trial in the parent run
        """
        trial_result = trials.best_trial["result"]
        classifier: ClassifierMixin = eval(trial_result["classifier"])()  # noqa: S307
        pipeline = self.create_pipeline(classifier)
        pipeline.set_params(**trial_result["parameters"])
        pipeline.fit(X, y)
        mlflow.log_params(pipeline.get_params())

        mlflow.sklearn.log_model(pipeline, "classifier")
        object_to_mlflow(self.label_encoder, "label_encoder")

        artifact_uri = Path(mlflow.active_run().info.artifact_uri)
        artifacts = {
            "model_path": (artifact_uri / "classifier").as_posix(),
            "label_encoder_path": (artifact_uri / "label_encoder").as_posix(),
        }
        input_example = [X[0]]
        path = Path(__file__).parents[1] / "deployable_mlflow_model.py"
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=path,
            artifacts=artifacts,
            input_example=input_example,
            registered_model_name=self.budget_id,
        )
        self._log_metrics_of_run(trial_result["run_id"])

        self.logger.info("Finished training")

    def _log_metrics_of_run(self, run_id: str) -> None:
        """Log the metrics of a run in the current run."""
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        metrics = run.data.metrics
        mlflow.log_metrics(metrics)
