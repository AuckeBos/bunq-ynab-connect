import os
from datetime import datetime
from logging import LoggerAdapter

import pandas as pd
import requests
from dateutil import parser
from kink import inject
from mlserver.codecs import PandasCodec
from ynab import TransactionDetail
from ynab.configuration import Configuration

from bunq_ynab_connect.clients.ynab_client import YnabClient
from bunq_ynab_connect.data.bunq_account_to_ynab_account_mapper import (
    BunqAccountToYnabAccountMapper,
)
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.bunq_account import BunqAccount
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab_account import YnabAccount
from bunq_ynab_connect.sync_bunq_to_ynab.payment_queue import PaymentQueue


class PaymentSyncer:
    """Class that syncs payments from Bunq to Ynab.

    Loops over the payment queue and syncs payments one by one.

    Attributes
    ----------
        logger: The logger to use to log messages.
        storage: The storage class to use to read the queue and payments from.
        client: The YNAB client to use to create transactions.
        mapper: The mapper to use to map Bunq accounts to YNAB accounts.
        queue: The queue to use to get the payments to sync.
        account_map: A map from Bunq account id to YNAB account.
        prediction_base_url: The url ofr the MLServerm odle to use to predict categories
            contains var "budget_id" to be replaced with the budget id.

    """

    FLAG_COLOR = "blue"
    CLEARING_STATUS = "uncleared"

    logger: LoggerAdapter
    storage: AbstractStorage
    client: YnabClient
    queue: PaymentQueue
    account_map: dict
    prediction_url: str

    @inject
    def __init__(  # noqa: PLR0913
        self,
        logger: LoggerAdapter,
        storage: AbstractStorage,
        client: YnabClient,
        mapper: BunqAccountToYnabAccountMapper,
        queue: PaymentQueue,
        mlserver_model_url: str,
    ):
        self.logger = logger
        self.storage = storage
        self.client = client
        self.mapper = mapper
        self.queue = queue
        self.account_map = mapper.map()
        self.prediction_url = mlserver_model_url

    def sanity_check_payment(self, payment: BunqPayment) -> bool:
        """Do a sanity check on the payment. Payment should not be synced if the fails.

        Check:
        - The 'date' should be larger than the START_SYNC_DATE from the env
            If the START_SYNC_DATE is not set, we raise an exception. In that case, we
            don't want to sync any payment, since we could overflow YNAB with old
            transactions.
        """
        min_date = os.getenv("START_SYNC_DATE")
        if min_date is None:
            msg = "START_SYNC_DATE is not set. Not syncing any payments"
            raise ValueError(msg)

        if payment.created < parser.parse(min_date):
            self.logger.warning(
                "Payment %s is older than %s. Not syncing", payment.id, min_date
            )
            return False
        return True

    def payment_to_transaction(
        self, payment: BunqPayment, account: YnabAccount
    ) -> TransactionDetail:
        """Create a YNAB transaction from a Bunq payment.

        Use values from the payment to fill in the transaction.

        Parameters
        ----------
            payment: The Bunq payment
            account: The YNAB account to which the payment belongs

        Returns
        -------
            The YNAB transaction to be created

        """
        name = (
            payment.counterparty_alias["display_name"]
            if "display_name" in payment.counterparty_alias
            else payment.counterparty_alias["name"]
        )
        configuration = Configuration()
        configuration.client_side_validation = False
        amount = int(float(payment.amount["value"]) * 1000)  # Convert to milliunits
        return TransactionDetail(
            account_id=account.id,
            amount=amount,
            date=payment.created,
            payee_name=name,
            memo=payment.description,
            cleared=self.CLEARING_STATUS,
            flag_color=self.FLAG_COLOR,
            approved=False,
            local_vars_configuration=configuration,
            category_name=self.decide_category(payment, account),
        )

    def decide_category(self, payment: BunqPayment, account: YnabAccount) -> str:
        """Decide the category of a payment. Use the ML server to predict the category.

        - Build endpoint
        - Convert payment to dict
             Note: Cannot use payment.dict(), because datetime is not serializable.
        - Use the PandasCodec to encode the request
        - Send the request to the ML server
        - Return the predicted category
        - Upon any failure, log failure and return None.
            In this case, Ynab will use the lastly used category for the payee.
        """
        try:
            invalid_categories = ["Split (Multiple Categories)..."]
            endpoint = self.prediction_url.format(budget_id=account.budget_id)
            response = requests.post(
                endpoint,
                data=PandasCodec.encode_request(
                    pd.DataFrame.from_dict([payment.model_dump()])
                )
                .model_dump_json()
                .encode(),
                timeout=10,
            )
            response.raise_for_status()
            prediction = response.json()["outputs"][0]["data"][0]
            if prediction in invalid_categories:
                msg = f"Invalid category predicted: {prediction}"
                self.logger.exception(msg)
                raise ValueError(msg)  # noqa: TRY301
            self.logger.info(
                "Predicted category %s for payment %s", prediction, payment.id
            )
        except Exception:
            self.logger.exception(
                "Could not predict category for payment %s", payment.id
            )
            return None
        else:
            return prediction

    def create_transaction(self, payment: BunqPayment, account: YnabAccount) -> None:
        """Create a YNAB transaction from a Bunq payment and create it in YNAB.

        If the payment fails the sanity check, do not create the transaction.

        Parameters
        ----------
            payment: The Bunq payment
            account: The YNAB account to which the payment belongs


        """
        transaction = self.payment_to_transaction(payment, account)
        if not self.sanity_check_payment(payment):
            return

        self.client.create_transaction(transaction, account.budget_id)

    def sync_payment(self, payment_id: int, *, skip_if_synced: bool = True) -> None:
        """Sync a payment from Bunq to YNAB.

        - Load the payment from the database
        - Find the corresponding YNAB account
        - Create a transaction.

        Parameters
        ----------
            payment_id: The id of the payment to sync
            skip_if_synced: If True, skip the payment if it has already been synced


        """
        payment = self.storage.find_one("bunq_payments", [("id", "eq", payment_id)])
        if payment is None:
            msg = f"Could not find payment with id {payment_id}"
            raise ValueError(msg)
        if self.queue.is_yet_synced(payment_id):
            if skip_if_synced:
                self.logger.info(
                    "Payment %s already synced at %s, skipping",
                    payment_id,
                    self.queue.synced_at(payment_id),
                )
                return
            self.logger.warning(
                "Payment %s already synced at %s, force syncing",
                payment_id,
                self.queue.synced_at(payment_id),
            )
        payment = self.storage.rows_to_entities([payment], BunqPayment)[0]
        account_id = payment.monetary_account_id
        if account_id not in self.account_map:
            self.logger.warning(
                "Could not find YNAB account for Bunq account %s, not syncing payment %s",  # noqa: E501
                account_id,
                payment_id,
            )
            return
        ynab_account: YnabAccount = self.account_map[account_id]
        self.create_transaction(payment, ynab_account)
        self.queue.mark_synced(payment_id)

    def sync(self) -> None:
        """Sync all payments in the queue from Bunq to YNAB."""
        counter = 0
        while self.queue:
            with self.queue.pop() as payment_id:
                self.sync_payment(payment_id)
            counter += 1
        self.logger.info("Synced %s payments", counter)

    def sync_account(
        self,
        iban: str,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> None:
        """Force sync all payments for a specific account. Even if already synced.

        Parameters
        ----------
            iban: The IBAN of the account to sync
            from_date: The date from which to sync payments
            to_date: The date until which to sync payments

        """
        if account := BunqAccount.by_iban(iban=iban):
            query = [("monetary_account_id", "eq", account.id)]
            if from_date:
                query.append(
                    ("created", "gte", from_date.strftime("%Y-%m-%d %H:%M:%S.%f"))
                )
            if to_date:
                query.append(
                    ("created", "lte", to_date.strftime("%Y-%m-%d %H:%M:%S.%f"))
                )
            self.logger.info("Querying payments for account %s, query: %s", iban, query)
            payments = self.storage.find("bunq_payments", query)
            self.logger.info("Found %s payments for account %s", len(payments), iban)
            for payment in payments:
                self.sync_payment(payment["id"], skip_if_synced=False)
        else:
            self.logger.exception("Could not find account with IBAN %s", iban)
