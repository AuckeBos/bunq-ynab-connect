from logging import LoggerAdapter

from kink import di
from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner

from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.data_extractors.bunq_payment_extractor import (
    BunqPaymentExtractor,
)
from bunq_ynab_connect.data.data_extractors.ynab_account_extractor import (
    YnabAccountExtractor,
)
from bunq_ynab_connect.data.data_extractors.ynab_budget_extractor import (
    YnabBudgetExtractor,
)


@task(task_run_name="{extractor_class.__name__}.extract()")
def extract_one(extractor_class: AbstractExtractor.__class__):
    extractor = extractor_class()
    extractor.extract()


@flow(validate_parameters=False, task_runner=ConcurrentTaskRunner())
def extract():
    """
    Run all extractors.
    Run BunqPaymentExtractor in paralel with YnabBudgetExtractor and YnabAccountExtractor
    """
    di[LoggerAdapter] = get_run_logger()
    extract_one.submit(BunqPaymentExtractor)
    extract_one(YnabBudgetExtractor)
    extract_one(YnabAccountExtractor)
