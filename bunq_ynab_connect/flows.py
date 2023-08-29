from ast import List
from logging import LoggerAdapter

from kink import di
from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner

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
from bunq_ynab_connect.sync_bunq_to_ynab.payment_syncer import PaymentSyncer


@task()
def run_extractors(extractor_classes: List[AbstractExtractor.__class__]):
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
    di[LoggerAdapter] = get_run_logger()
    run_extractors.submit(BunqAccountExtractor, BunqPaymentExtractor)
    run_extractors.submit(
        YnabBudgetExtractor, YnabAccountExtractor, YnabTransactionExtractor
    )


@flow
def sync_payments():
    """
    Sync payments from bunq to YNAB.
    """
    syncer = PaymentSyncer()
    syncer.sync()
