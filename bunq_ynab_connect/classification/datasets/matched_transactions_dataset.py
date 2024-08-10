from logging import LoggerAdapter

from kink import inject

from bunq_ynab_connect.classification.datasets.abstract_dataset import AbstractDataset
from bunq_ynab_connect.data.bunq_account_to_ynab_account_mapper import (
    BunqAccountToYnabAccountMapper,
)
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction


class MatchedTransactionsDataset(AbstractDataset):
    """A dataset that contains the matched transactions between bunq and YNAB.

    Does not do any transformations, but simply matches and stores the full info.
    A model that uses the set should extract relevant features from the data.

    Attributes
    ----------
        NAME: The name of the dataset.
        KEY_COLUMN: The name of the column that is used as a key.
        map: A map of bunq account ids to ynab accounts, uses the
            BunqAccountToYnabAccountMapper
        MAX_ALLOWED_DIFF_FOR_MATCH: The maximum allowed difference between the amounts

    """

    NAME: str = "matched_transactions"
    KEY_COLUMN: str = "id"
    map: dict
    MAX_ALLOWED_DIFF_FOR_MATCH = 0.05

    @inject
    def __init__(
        self,
        mapper: BunqAccountToYnabAccountMapper,
        storage: AbstractStorage,
        logger: LoggerAdapter,
    ) -> None:
        super().__init__(storage, logger)
        self.map = mapper.map()

    def is_match(
        self,
        bunq_payment: BunqPayment,
        ynab_transaction: YnabTransaction,
    ) -> bool:
        """Check if a bunq payment and a ynab transaction are a match.

        - The dates are equal
        - The amounts are equal
        """
        return bunq_payment.created.date() == ynab_transaction.date.date() and float(
            bunq_payment.amount["value"]
        ) == round(ynab_transaction.amount / 1000, 2)

    def match(self, bunq_payments: list, ynab_transactions: list) -> list:
        """Match the bunq payments to the ynab transactions.

        Returns
        -------
            A list of matches. A match is:
            {
                "match_id": ynab_transaction.id, # Identifies the row
                "bunq_payment": BunqPayment,
                "ynab_transaction": YnabTransaction
            }

        """
        matches = []
        for ynab_transaction in ynab_transactions:
            if not self.sanity_check_ynab_transaction(ynab_transaction):
                continue
            for bunq_payment in bunq_payments:
                if not self.sanity_check_bunq_payment(bunq_payment):
                    continue
                if self.is_match(bunq_payment, ynab_transaction):
                    matches.append(
                        {
                            "match_id": ynab_transaction.id,
                            "bunq_payment": bunq_payment.dict(),
                            "ynab_transaction": ynab_transaction.dict(),
                        }
                    )
                    break
        return matches

    def load_candidates(self, bunq_account_id: int, ynab_account_id: int) -> tuple:
        """Load the candidates for matching.

        - Load all bunq payments for account with created date after the last runmoment
        - Load all ynab transactions for account with date after the last runmoment

        Returns
        -------
            A tuple with the bunq payments and ynab transactions

        """
        bunq_payments = self.storage.find(
            "bunq_payments",
            [
                ("monetary_account_id", "eq", bunq_account_id),
                ("created", "gte", self.last_runmoment.isoformat()),
            ],
        )
        ynab_transactions = self.storage.find(
            "ynab_transactions",
            [
                ("account_id", "eq", ynab_account_id),
                ("date", "gte", self.last_runmoment),
            ],
        )
        bunq_payments = self.storage.rows_to_entities(bunq_payments, BunqPayment)
        ynab_transactions = self.storage.rows_to_entities(
            ynab_transactions, YnabTransaction
        )
        return bunq_payments, ynab_transactions

    def load_new_data(self) -> list:
        """Load the new data from the storage.

        - For each matched Bunq and YNAB account
        - Load all bunq payments and ynab transactions
        - Match the bunq payments to the ynab transactions
        """
        matches = []
        for bunq_account_id, ynab_account in self.map.items():
            ynab_account_id = ynab_account.id
            bunq_payments, ynab_transactions = self.load_candidates(
                bunq_account_id, ynab_account_id
            )
            current_matches = self.match(bunq_payments, ynab_transactions)
            if len(current_matches) > 0:
                self.logger.info(
                    "Found %s matches for %s",
                    len(current_matches),
                    ynab_account.name,
                )
            matches.extend(current_matches)
        return matches

    def sanity_check_ynab_transaction(self, ynab_transaction: YnabTransaction) -> bool:
        """Sanity check for whether a ynab transaction should be included.

        - Check that the amount is at least +=0.05. Lower amounts are test payments
        """
        return abs(ynab_transaction.amount) > self.MAX_ALLOWED_DIFF_FOR_MATCH

    def sanity_check_bunq_payment(self, bunq_payment: BunqPayment) -> bool:
        """Sanity check for whether a bunq payment should be included in the dataset.

        - Check that the amount is at least +=0.05. Lower amounts are test payments
        """
        return (
            abs(float(bunq_payment.amount["value"])) > self.MAX_ALLOWED_DIFF_FOR_MATCH
        )
