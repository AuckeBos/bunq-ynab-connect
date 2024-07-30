import click
from kink import inject

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
def cli() -> None:
    """Provide the cli."""


@cli.command()
def extract() -> None:
    """Run all extractors."""
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
def sync_payments() -> None:
    """Sync payments from bunq to YNAB."""
    syncer = PaymentSyncer()
    syncer.sync_queue()


@cli.command()
@click.argument("payment_id", type=int)
def sync_payment(payment_id: int) -> None:
    """Sync a single payment from bunq to YNAB."""
    syncer = PaymentSyncer()
    syncer.sync_payment(payment_id)


@cli.command()
@inject
def test(storage: AbstractStorage) -> None:  # noqa: ARG001
    """Testing function."""
    trainer = Trainer(budget_id="my-budget")
    trainer.train()


@cli.command()
def help() -> None:  # noqa: A001
    """Show the help message."""
    ctx = click.Context(cli)
    click.echo(ctx.get_help())


if __name__ == "__main__":
    test()
