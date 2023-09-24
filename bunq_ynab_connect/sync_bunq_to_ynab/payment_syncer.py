import json
import os
from logging import LoggerAdapter

import pandas as pd
import requests
from bunq.sdk.model.generated.endpoint import (
    BunqResponsePaymentList,
    MonetaryAccountBank,
    Payment,
)
from bunq.sdk.model.generated.object_ import MonetaryAccountReference
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
from bunq_ynab_connect.models.bunq_payment import BunqPayment
from bunq_ynab_connect.models.ynab_account import YnabAccount
from bunq_ynab_connect.sync_bunq_to_ynab.payment_queue import PaymentQueue


class PaymentSyncer:
    """
    Class that syncs payments from Bunq to Ynab.

    Loops over the payment queue and syncs payments one by one.

    Attributes:
        logger: The logger to use to log messages.
        storage: The storage class to use to read the queue and payments from.
        client: The YNAB client to use to create transactions.
        mapper: The mapper to use to map Bunq accounts to YNAB accounts.
        queue: The queue to use to get the payments to sync.
        account_map: A map from Bunq account id to YNAB account.
        prediction_base_url: The base url of the ML server to use to predict categories. Read from env.

    """

    FLAG_COLOR = "blue"
    CLEARING_STATUS = "uncleared"

    logger: LoggerAdapter
    storage: AbstractStorage
    client: YnabClient
    queue: PaymentQueue
    account_map: dict
    prediction_base_url: str

    @inject
    def __init__(
        self,
        logger: LoggerAdapter,
        storage: AbstractStorage,
        client: YnabClient,
        mapper: BunqAccountToYnabAccountMapper,
        queue: PaymentQueue,
    ):
        self.logger = logger
        self.storage = storage
        self.client = client
        self.mapper = mapper
        self.queue = queue
        self.account_map = mapper.map()
        self.prediction_base_url = f"{os.getenv('MLSERVER_URL')}/v2/models"

    def sanity_check_payment(self, payment: BunqPayment) -> bool:
        """
        Do a sanity check on the payment. Payment should not be synced if the check fails.
        Check:
        - The 'date' should be larger than the START_SYNC_DATE from the env
            If the START_SYNC_DATE is not set, we raise an exception. In that case, we
            don't want to sync any payment, since we could overflow YNAB with old transactions.
        """
        min_date = os.getenv("START_SYNC_DATE")
        if min_date is None:
            raise ValueError("START_SYNC_DATE is not set. Not syncing any payments")

        if payment.created < parser.parse(min_date):
            self.logger.warning(
                f"Payment {payment.id} is older than {min_date}. Not syncing"
            )
            return False
        return True

    def payment_to_transaction(
        self, payment: BunqPayment, account: YnabAccount
    ) -> TransactionDetail:
        """
        Create a YNAB transaction from a Bunq payment.
        Use values from the payment to fill in the transaction.

        Args:
            payment: The Bunq payment
            account: The YNAB account to which the payment belongs

        Returns:
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
        transaction = TransactionDetail(
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
        return transaction

    def decide_category(self, payment: BunqPayment, account: YnabAccount) -> str:
        """
        Decide the category of a payment. Use the ML server to predict the category.

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
            endpoint = f"{self.prediction_base_url}/{account.budget_id}/infer"
            payment_as_dict = json.loads(payment.json())
            request = PandasCodec.encode_request(
                pd.DataFrame.from_dict([payment_as_dict])
            )
            response = requests.post(endpoint, json=request.dict())
            if response.status_code != 200:
                raise ValueError(f"Error in response: {response.text}")
            prediction = response.json()["outputs"][0]["data"][0]
            if prediction in invalid_categories:
                raise ValueError(f"Invalid category predicted: {prediction}")
            self.logger.info(
                f"Predicted category {prediction} for payment {payment.id}"
            )
            return prediction
        except Exception as e:
            self.logger.error(
                f"Could not predict category for payment {payment.id}: {e}. Falling back to None"
            )
            return None

    def create_transaction(self, payment: BunqPayment, account: YnabAccount):
        """
        Create a YNAB transaction from a Bunq payment and create it in YNAB.
        If the payment fails the sanity check, do not create the transaction.

        Args:
            payment: The Bunq payment
            account: The YNAB account to which the payment belongs

        """
        transaction = self.payment_to_transaction(payment, account)
        if not self.sanity_check_payment(payment):
            return

        self.client.create_transaction(transaction, account.budget_id)

    def sync_payment(self, payment_id: int):
        """
        Sync a payment from Bunq to YNAB.
        - Load the payment from the database
        - Find the corresponding YNAB account
        - Create a transaction.

        Args:
            payment_id: The id of the payment to sync

        """
        payment = self.storage.find_one("bunq_payments", [("id", "eq", payment_id)])
        if payment is None:
            raise ValueError(f"Could not find payment with id {payment_id}")
        payment = self.storage.rows_to_entities([payment], BunqPayment)[0]
        account_id = payment.monetary_account_id
        if account_id not in self.account_map:
            self.logger.warning(
                f"Could not find YNAB account for Bunq account {account_id}. Not syncing payment {payment_id}"
            )
            return
        ynab_account: YnabAccount = self.account_map[account_id]
        self.create_transaction(payment, ynab_account)

    def sync(self):
        """
        Sync all payments in the queue from Bunq to YNAB.
        """
        counter = 0
        while self.queue:
            with self.queue.pop() as payment_id:
                self.sync_payment(payment_id)
            counter += 1
        self.logger.info(f"Synced {counter} payments")
