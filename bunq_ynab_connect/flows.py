from kink import di
from prefect import flow, task
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
from bunq_ynab_connect.sync_bunq_to_ynab.payment_syncer import PaymentSyncer


@task()
def run_extractors(extractor_classes: list):
    """
    Run the provided extractors serially
    """
    for extractor_class in extractor_classes:
        extractor = extractor_class()
        extractor.extract()


@flow(validate_parameters=False)
def extract():
    """
    Run all extractors.
    Run Bunq and YNAB extractors in parallel.
    """
    extractors = [
        BunqAccountExtractor(),
        BunqPaymentExtractor(),
        YnabBudgetExtractor(),
        YnabAccountExtractor(),
        YnabTransactionExtractor(),
    ]
    for extractor in extractors:
        extractor.extract()


@flow
def sync_payment_queue():
    """
    Sync all payements in the payment queue.
    """
    syncer = PaymentSyncer()
    syncer.sync()


@flow
def sync_payment(payment_id: int):
    """
    Sync a single payment from bunq to YNAB.
    """
    syncer = PaymentSyncer()
    syncer.sync_payment(payment_id)


@flow
def sync():
    """
    Run all extractors and sync payments.
    """
    extract()
    sync_payment_queue()


@task(task_run_name="train_for_budget_{budget_id}")
def train_for_budget(budget_id: str):
    """
    Run the trainer for a single budget.
    Deploy the model after training.
    """
    trainer = Trainer(budget_id=budget_id)
    run_id = trainer.train()
    if run_id:
        deployer = Deployer(budget_id=budget_id)
        deployer.deploy(run_id)


@flow(task_runner=DaskTaskRunner())
def train():
    """
    Train one classifier for each budget.
    Before training, update the feature store, to make sure the latest data is used.
    """
    feature_store = FeatureStore()
    feature_store.update()

    storage = di[AbstractStorage]
    budget_ids = storage.get_budget_ids()
    for budget_id in budget_ids:
        train_for_budget.submit(budget_id)


@flow
def exchange_pat(pat: str):
    """
    Exchange a new PAT for a new API context.
    """
    client = di[BunqClient]
    client.exchange_pat(pat)
