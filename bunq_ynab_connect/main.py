import click
from kink import inject

from bunq_ynab_connect.classification.deployer import Deployer
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
    syncer.sync()


@cli.command()
@click.argument("payment_id", type=int)
@click.argument("skip_if_synced", type=bool, default=True)
def sync_payment(payment_id: int, skip_if_synced: bool) -> None:  # noqa: FBT001
    """Sync a single payment from bunq to YNAB."""
    syncer = PaymentSyncer()
    syncer.sync_payment(payment_id, skip_if_synced=skip_if_synced)


@cli.command()
@inject
def test(storage: AbstractStorage) -> None:  # noqa: ARG001
    """Testing function."""
    budget_id = "my-budget"
    trainer = Trainer(budget_id=budget_id)
    trainer.train()
    run_id = trainer.train()
    if run_id:
        deployer = Deployer(budget_id=budget_id)
        deployer.deploy(run_id)


@cli.command()
def help() -> None:  # noqa: A001
    """Show the help message."""
    ctx = click.Context(cli)
    click.echo(ctx.get_help())


if __name__ == "__main__":
    test()
