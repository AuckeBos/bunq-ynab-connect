import click
from kink import inject
from sklearn.tree import DecisionTreeClassifier

from bunq_ynab_connect.classification.datasets.matched_transactions_dataset import (
    MatchedTransactionsDataset,
)
from bunq_ynab_connect.classification.experiments.classifier_selection_experiment import (
    ClassifierSelectionExperiment,
)
from bunq_ynab_connect.classification.experiments.classifier_tuning_experiment import (
    ClassifierTuningExperiment,
)
from bunq_ynab_connect.classification.feature_store import FeatureStore
from bunq_ynab_connect.classification.trainer import Trainer
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


@click.group
def cli():
    """
    Main cli
    """


@cli.command()
def extract():
    """
    Run all extractors.
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


@cli.command()
def sync_payments():
    """
    Sync payments from bunq to YNAB.
    """
    syncer = PaymentSyncer()
    syncer.sync()


@cli.command()
@click.argument("payment_id", type=int)
def sync_payment(payment_id: int):
    """
    Sync a single payment from bunq to YNAB.
    """
    syncer = PaymentSyncer()
    syncer.sync_payment(payment_id)


@cli.command()
@inject
def test(storage: AbstractStorage):
    """
    Testing function
    """
    trainer = Trainer()
    trainer.train("my-budget")


@cli.command()
def help():
    """
    Show the help message.
    """
    ctx = click.Context(cli)
    click.echo(ctx.get_help())


if __name__ == "__main__":
    test()
