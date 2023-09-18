from kink import di
from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner
from prefect_dask.task_runners import DaskTaskRunner

from bunq_ynab_connect.classification.deployer import Deployer
from bunq_ynab_connect.classification.experiments.classifier_selection_experiment import (
    ClassifierSelectionExperiment,
)
from bunq_ynab_connect.classification.trainer import Trainer
from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
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


@flow(validate_parameters=False, task_runner=ConcurrentTaskRunner())
def extract():
    """
    Run all extractors.
    Run Bunq and YNAB extractors in parallel.
    """
    run_extractors.submit([BunqAccountExtractor, BunqPaymentExtractor])
    run_extractors.submit(
        [YnabBudgetExtractor, YnabAccountExtractor, YnabTransactionExtractor]
    )


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


@task(task_run_name="train_for_budget{budget_id}")
def train_for_budget(budget_id: str, threads: int = 1):
    """
    Run the trainer for a single budget.
    Set threads to 1, since we use prefect to parallelize the training.
    Deploy the model after training.
    """
    trainer = Trainer(budget_id=budget_id, threads=threads)
    run_id = trainer.train()
    if run_id:
        deployer = Deployer(budget_id=budget_id)
        deployer.deploy(run_id)


@flow(task_runner=DaskTaskRunner())
def train(threads: int = 1):
    """
    Train one classifier for each budget
    """
    storage = di[AbstractStorage]
    budget_ids = storage.get_budget_ids()
    for budget_id in budget_ids:
        train_for_budget.submit(budget_id, threads)
