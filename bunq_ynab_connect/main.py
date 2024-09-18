import click
from kink import inject
from sqlalchemy import Engine
from sqlmodel import Session

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
from bunq_ynab_connect.models.bunq_account import BunqAccount
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
def test(storage: AbstractStorage, database: Engine) -> None:  # noqa: ARG001
    """Testing function."""
    from sqlmodel import select

    with Session(database) as session:
        accounts = session.exec(select(BunqAccount)).all()
        # note: only works with open session
        aliasses = accounts[0].aliasses
        test = ""
    BunqAccountExtractor().extract()

    # trainer = Trainer(budget_id="my-budget")
    # trainer.train()


@cli.command()
def help() -> None:  # noqa: A001
    """Show the help message."""
    ctx = click.Context(cli)
    click.echo(ctx.get_help())


if __name__ == "__main__":
    test()
