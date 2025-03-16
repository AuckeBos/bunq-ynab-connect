from datetime import datetime
from typing import TYPE_CHECKING

from kink import di
from prefect import flow, serve, task
from prefect.client.schemas.schedules import CronSchedule
from prefect.concurrency.sync import concurrency
from prefect_dask.task_runners import DaskTaskRunner

from bunq_ynab_connect.classification.deployer import Deployer
from bunq_ynab_connect.classification.feature_store import FeatureStore
from bunq_ynab_connect.classification.trainer import Trainer
from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.data.data_extractors.bunq_account_extractor import (
    BunqAccountExtractor,
)
from bunq_ynab_connect.data.data_extractors.bunq_payment_extractor import (
    BunqPaymentExtractor,
)
from bunq_ynab_connect.data.data_extractors.ynab_account_extractor import (
    YnabAccountExtractor,
)
from bunq_ynab_connect.data.data_extractors.ynab_budget_extractor import (
    YnabBudgetExtractor,
)
from bunq_ynab_connect.data.data_extractors.ynab_transaction_extractor import (
    YnabTransactionExtractor,
)
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab_budget import YnabBudget
from bunq_ynab_connect.sync_bunq_to_ynab.payment_syncer import PaymentSyncer

if TYPE_CHECKING:
    from bunq_ynab_connect.data.data_extractors.abstract_extractor import (
        AbstractExtractor,
    )


@flow
def extract() -> None:
    """Run all extractors.

    Run Bunq and YNAB extractors in parallel.
    """
    extractors: list[AbstractExtractor] = [
        BunqAccountExtractor(),
        BunqPaymentExtractor(),
        YnabBudgetExtractor(),
        YnabAccountExtractor(),
        YnabTransactionExtractor(),
    ]
    for extractor in extractors:
        extractor.extract()


@flow
def sync_payment_queue() -> None:
    """Sync all payements in the payment queue."""
    syncer = PaymentSyncer()
    syncer.sync()


@flow
def sync_payment(payment_id: int, skip_if_synced: bool) -> None:  # noqa: FBT001
    """Sync a single payment from bunq to YNAB."""
    with concurrency("single-payment-sync", timeout_seconds=60):
        syncer = PaymentSyncer()
        syncer.sync_payment(payment_id, skip_if_synced=skip_if_synced)


@flow
def sync() -> None:
    """Run all extractors and sync payments."""
    extract()
    sync_payment_queue()


@flow
def sync_payments_of_account(
    iban: str, from_date: datetime | None = None, to_date: datetime | None = None
) -> None:
    """Sync payments for a single IBAN."""
    syncer = PaymentSyncer()
    syncer.sync_account(iban, from_date, to_date)


@task(task_run_name="train_for_budget_{budget_id}")
def train_for_budget(budget_id: str, max_runs: int) -> None:
    """Run the trainer for a single budget.

    Deploy the model after training.
    """
    trainer = Trainer(budget_id=budget_id, max_runs=max_runs)
    run_id = trainer.train()
    if run_id:
        deployer = Deployer(budget_id=budget_id)
        deployer.deploy(run_id)


@flow(task_runner=DaskTaskRunner())
def train(max_runs: int = 250) -> None:
    """Train one classifier for each budget.

    Before training, update the feature store, to make sure the latest data is used.

    Parameters
    ----------
    max_runs : int
        Maximum number of runs for hyperopt.

    """
    feature_store = FeatureStore()
    feature_store.update()

    storage = di[AbstractStorage]
    budget_ids = YnabBudget.get_budget_ids(storage)
    for budget_id in budget_ids:
        train_for_budget.submit(budget_id=budget_id, max_runs=max_runs)


@flow
def exchange_pat(pat: str, name: str) -> None:
    """Exchange a new PAT for a new API context."""
    client = BunqClient()
    client.exchange_pat(pat, name)


def work() -> None:
    """Create a deployment for each flow, and serve all of them."""
    serve(
        sync.to_deployment(
            name="sync",
            description="Extract all data and sync to Ynab",
            version="2024.09.12",
            schedules=[
                CronSchedule(
                    cron="00 06-23 * * *", timezone="Europe/Amsterdam", day_or=True
                )
            ],
        ),
        extract.to_deployment(
            name="extract",
            description="Extract Bunq & Ynab data",
            version="2024.09.12",
        ),
        sync_payment_queue.to_deployment(
            name="sync_payment_queue",
            description="Sync all queued payments to Ynab",
            version="2024.09.12",
        ),
        sync_payment.to_deployment(
            name="sync_payment",
            description="Sync a single payment to Ynab",
            version="2024.09.12",
            parameters={"skip_if_synced": True},
        ),
        sync_payments_of_account.to_deployment(
            name="sync_payments_of_account",
            description="Force sync payemnts of one iban within a date range",
            version="2024.09.12",
        ),
        train.to_deployment(
            name="train",
            description="Train one classifier for each budget",
            version="2024.09.12",
            schedules=[CronSchedule(cron="0 2 * * 0", timezone="Europe/Amsterdam")],
            parameters={"max_runs": 100},
        ),
        exchange_pat.to_deployment(
            name="exchange_pat",
            description="Exchange a PAT for a Bunq config file",
            version="2024.09.12",
        ),
    )
