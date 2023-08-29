from logging import LoggerAdapter

from bunq.sdk.model.generated.endpoint import (
    BunqResponsePaymentList,
    MonetaryAccountBank,
    Payment,
)
from kink import inject

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.ynab.bunq_account import BunqAccount
from bunq_ynab_connect.models.ynab.ynab_account import YnabAccount


@inject
class BunqAccountToYnabAccountMapper:
    """
    Class that maps a Bunq account to a YNAB Account.
    Assume that the "notes" field of a YNAB account equals the IBAN to which it belongs
    """

    @inject
    def __init__(self, storage: AbstractStorage, logger: LoggerAdapter):
        self.storage = storage
        self.logger = logger

    def map(self) -> dict[str, YnabAccount]:
        """
        Map the Bunq accounts to the YNAB accounts.

        Returns:
            A dict with the Bunq account id as key and the YNAB account as value
        """
        bunq_accounts = self.storage.get_as_entity("bunq_accounts", BunqAccount, False)
        ynab_accounts = self.storage.get_as_entity("ynab_accounts", YnabAccount, False)
        bunq_iban_to_ynab_account = {}
        for bunq_account in bunq_accounts:
            for ynab_account in ynab_accounts:
                if ynab_account.note == bunq_account.iban:
                    bunq_iban_to_ynab_account[bunq_account.id] = ynab_account
                    break
        return bunq_iban_to_ynab_account
