import click

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


@cli.command
def test():
    """
    Testing function
    """
    syncer = PaymentSyncer()
    syncer.sync()


@cli.command
def help():
    """
    Show the help message.
    """
    ctx = click.Context(cli)
    click.echo(ctx.get_help())


if __name__ == "__main__":
    test()
