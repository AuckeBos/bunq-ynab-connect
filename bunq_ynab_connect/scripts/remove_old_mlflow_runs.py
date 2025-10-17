import logging
from datetime import datetime, timedelta, timezone
from logging import LoggerAdapter

import mlflow
from kink import di
from mlflow.store.artifact.artifact_repository_registry import (
    get_artifact_repository,
)
from mlflow.tracking import MlflowClient

from bunq_ynab_connect.helpers.general import now

logging.basicConfig(level=logging.INFO)

DAYS_TO_KEEP = 30


def run() -> None:
    """Remove MLFLOW runs older than 30 days."""
    # setup
    client = MlflowClient()
    cutoff_date = now() - timedelta(days=DAYS_TO_KEEP)
    logger = di[LoggerAdapter]
    logger.debug(
        "Deleting runs older than %d days (before %s)",
        DAYS_TO_KEEP,
        cutoff_date.isoformat(),
    )
    logger.debug("Tracking URI: %s", mlflow.get_tracking_uri())

    # Loop over experiments
    experiments = client.search_experiments()
    for exp in experiments:
        logger.debug("Checking experiment: %s (%s)", exp.name, exp.experiment_id)

        # Fetch all runs
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=10000,
        )

        for run in runs:
            start_time = datetime.fromtimestamp(
                run.info.start_time / 1000, tz=timezone.utc
            )

            if start_time < cutoff_date:
                run_id = run.info.run_id
                logger.debug(
                    "Deleting run from experiment %s (start: %s)",
                    exp.name,
                    start_time.isoformat(),
                )

                # Delete run metadata from tracking server
                client.delete_run(run_id)

                # Delete artifacts
                artifact_uri: str = run.info.artifact_uri
                logger.debug("Deleting artifacts at %s", artifact_uri)
                repo = get_artifact_repository(artifact_uri)
                repo.delete_artifacts()

    logger.debug("Cleanup complete.")
