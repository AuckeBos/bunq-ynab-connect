from logging import LoggerAdapter

from kink import inject
from sklearn.base import ClassifierMixin

import mlflow
from bunq_ynab_connect.classification.experiments.classifier_selection_experiment import (
    ClassifierSelectionExperiment,
)
from bunq_ynab_connect.classification.experiments.classifier_tuning_experiment import (
    ClassifierTuningExperiment,
)
from bunq_ynab_connect.classification.experiments.full_training_experiment import (
    FullTrainingExperiment,
)


class Trainer:
    """
    Train a classifier.
    Use the ClassifierSelectionExperiment to select the best classifier.
    Use the ClassifierTuningExperiment to select the best parameters for the given classifier.
    Train the best classifier on the full dataset.

    Attributes
        EXPERIMENT_NAME: Name of the experiment
        logger: LoggerAdapter
        budget_id: ID of the budget to train the classifier for
    """

    EXPERIMENT_NAME = "Full Training"

    logger: LoggerAdapter
    budget_id: str

    @inject
    def __init__(self, logger: LoggerAdapter, budget_id: str):
        self.logger = logger
        self.budget_id = budget_id

    budget_id: str

    def train(self) -> str:
        """
        Select the best classifier and the best parameters for the given classifier.
        Then train the best classifier on the full dataset.

        Returns
            The run ID of the FullTrainingExperiment
        """
        self.logger.info(f"Training for budget {self.budget_id}")
        classifier = self.select_best_classifier()
        if not classifier:
            self.logger.info("No classifier selected, training failed")
            return None
        parameters = self.select_best_parameters(classifier)
        return self.train_classifier(classifier, parameters)

    def select_best_classifier(self) -> ClassifierMixin:
        """
        Run the ClassifierSelectionExperiment to select the best classifier.
        """
        experiment = ClassifierSelectionExperiment(budget_id=self.budget_id)
        experiment.run()
        return experiment.get_best_classifier()

    def select_best_parameters(self, classifier) -> dict:
        """
        Run the ClassifierTuningExperiment to select the best parameters for the given classifier.
        """
        experiment = ClassifierTuningExperiment(
            budget_id=self.budget_id, clf=classifier
        )
        experiment.run()
        return experiment.get_best_parameters()

    def train_classifier(self, classifier, parameters) -> str:
        """
        Run the FullTrainingExperiment to train the classifier on the full dataset.

        Returns
            The run ID of the experiment
        """
        experiment = FullTrainingExperiment(
            budget_id=self.budget_id, clf=classifier, parameters=parameters
        )
        experiment.run()
        return experiment.parent_run_id
