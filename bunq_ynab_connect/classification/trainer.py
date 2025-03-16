from logging import LoggerAdapter

from kink import inject

from bunq_ynab_connect.classification.experiments.find_best_model_experiment import (
    FindBestModelExperiment,
)


class Trainer:
    """Train a classifier.

    Finds the best model (config) and trains it. Log to mlflow.

    Attributes
    ----------
        logger: LoggerAdapter
        budget_id: ID of the budget to train the classifier for
        max_runs: Maximum number of runs for hyperopt

    """

    logger: LoggerAdapter
    budget_id: str
    max_runs: int

    @inject
    def __init__(self, logger: LoggerAdapter, budget_id: str, max_runs: int):
        self.logger = logger
        self.budget_id = budget_id
        self.max_runs = max_runs

    def train(self) -> str:
        """Train the classifier by running an experiment.

        The experiment must find the best model (configuration), and
        finally train on the complete dataset and log it. It should be logged
        as a DeployableMlflowModel.

        Returns
        -------
            The run ID of the experiment. It serves as input to the
            Deployer

        """
        self.logger.info(
            "Training for budget %s (max %s runs)", self.budget_id, self.max_runs
        )

        experiment = FindBestModelExperiment(budget_id=self.budget_id)
        experiment.max_runs = self.max_runs
        experiment.run()
        return experiment.parent_run_id
