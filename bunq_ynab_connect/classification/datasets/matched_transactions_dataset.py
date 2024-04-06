from logging import LoggerAdapter

from bunq_ynab_connect.classification.datasets.abstract_dataset import AbstractDataset
from bunq_ynab_connect.data.bunq_account_to_ynab_account_mapper import (
    BunqAccountToYnabAccountMapper,
)
from bunq_ynab_connect.data.bunq_payment_to_ynab_transaction_mapper import (
    PaymentTransactionMapper,
)
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab_transaction import YnabTransaction
from kink import inject


class MatchedTransactionsDataset(AbstractDataset):
    """
    A dataset that contains the matched transactions between bunq and YNAB.
    Does not do any transformations, but simply matches and stores the full info.
    A model that uses the set should extract relevant features from the data.

    Attributes:
        NAME: The name of the dataset.
        KEY_COLUMN: The name of the column that is used as a key.
        map: A map of bunq account ids to ynab accounts, uses the
            BunqAccountToYnabAccountMapper
        payment_mapper: A PaymentTransactionMapper instance to match payments
    """

    NAME: str = "matched_transactions"
    KEY_COLUMN: str = "id"
    account_map: dict
    payment_mapper: PaymentTransactionMapper

    @inject
    def __init__(
        self,
        account_mapper: BunqAccountToYnabAccountMapper,
        payment_mapper: PaymentTransactionMapper,
        storage: AbstractStorage,
        logger: LoggerAdapter,
    ):
        super().__init__(storage, logger)
        self.payment_mapper = payment_mapper
        self.account_map = account_mapper.map()

    def _load_candidates(self, bunq_account_id: int, ynab_account_id: int) -> tuple:
        """
        Load the candidates for matching.
        - Load all bunq payments for the account with a created date > last runmoment
        - Load all ynab transactions for the account with a date > last runmoment

        Returns:
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
            ynab_transactions,
            YnabTransaction,
        )
        return bunq_payments, ynab_transactions

    def load_new_data(self) -> list:
        """
        Load the new data from the storage:
        - For each matched Bunq and YNAB account
        - Load all bunq payments and ynab transactions
        - Match the bunq payments to the ynab transactions
        """
        matches = []
        for bunq_account_id, ynab_account in self.account_map.items():
            ynab_account_id = ynab_account.id
            bunq_payments, ynab_transactions = self._load_candidates(
                bunq_account_id,
                ynab_account_id,
            )
            current_matches = self.payment_mapper.map(bunq_payments, ynab_transactions)
            if len(current_matches) > 0:
                self.logger.info(
                    f"Found {len(current_matches)} matches for {ynab_account.name}",
                )
            matches.extend(current_matches)
        return matches
