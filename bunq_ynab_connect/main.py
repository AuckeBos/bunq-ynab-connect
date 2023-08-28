import click

from bunq_ynab_connect.data.data_extractors.bunq_payment_extractor import (
    BunqPaymentExtractor,
)
from bunq_ynab_connect.data.data_extractors.ynab_account_extractor import (
    YnabAccountExtractor,
)
from bunq_ynab_connect.data.data_extractors.ynab_budget_extractor import (
    YnabBudgetExtractor,
)


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
    extractors = [BunqPaymentExtractor(), YnabBudgetExtractor(), YnabAccountExtractor()]
    for extractor in extractors:
        extractor.extract()


@cli.command
def test():
    """
    Testing function
    """
    extractors = [YnabBudgetExtractor(), YnabAccountExtractor()]
    for extractor in extractors:
        extractor.extract()


@cli.command
def help():
    """
    Show the help message.
    """
    ctx = click.Context(cli)
    click.echo(ctx.get_help())
