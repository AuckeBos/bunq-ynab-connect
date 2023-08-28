import os
from datetime import datetime
from logging import LoggerAdapter
from typing import List

from bunq import ApiEnvironmentType, Pagination
from bunq.sdk.context.api_context import ApiContext
from bunq.sdk.context.bunq_context import BunqContext
from bunq.sdk.model.generated import endpoint
from bunq.sdk.model.generated.endpoint import (
    BunqResponsePaymentList,
    MonetaryAccountBank,
    Payment,
)
from dateutil.parser import parse
from kink import inject

from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.config import BUNQ_CONFIG_FILE
from bunq_ynab_connect.helpers.general import cache, get_public_ip


@inject
class BunqClient:
    """
    Extractor for bunq payments.
    Loads all payments from all accounts.

    Attributes:
        storage: The storage to use
        logger: The logger to use
        PAYMENTS_PER_PAGE: The amount of payments to load per page
    """

    storage: AbstractStorage
    logger: LoggerAdapter
    PAYMENTS_PER_PAGE = 50

    @inject
    def __init__(self, storage: AbstractStorage, logger: LoggerAdapter) -> None:
        self.storage = storage
        self.logger = logger
        self._load_api_context()

    def _load_api_context(self):
        """
        Initialize context, ran on init
        - Check if bunq config file exists, if not, create it
        - Create ApiContext from bunq config file
        """
        self._check_api_context()
        context = ApiContext.restore(BUNQ_CONFIG_FILE)
        context.ensure_session_active()
        context.save(BUNQ_CONFIG_FILE)
        BunqContext.load_api_context(context)

    def _check_api_context(self):
        """
        Check if the bunq config file exists, if not, create it.
        """
        if not os.path.isfile(BUNQ_CONFIG_FILE):
            self.logger.info("Trading onetime token for bunq config file")
            onetime_token = os.getenv("BUNQ_ONETIME_TOKEN")
            if not onetime_token:
                self.logger.error("No bunq onetime token found")
                raise Exception(
                    "Please set your your bunq API key as BUNQ_ONETIME_TOKEN"
                )
            description = "bunq_ynab_connect"

            env = ApiEnvironmentType.PRODUCTION
            ips = [get_public_ip()]

            try:
                context = ApiContext.create(env, onetime_token, description, ips)
                context.save(BUNQ_CONFIG_FILE)
            except Exception as e:
                self.logger.error(f"Could not create _bunq config: {e}")
                raise Exception(f"Could not create _bunq config: {e}")
            self.logger.info("Created bunq config file")
        else:
            self.logger.info("Found bunq config file")

    def _should_continue_loading_payments(
        self,
        payment_list_response: BunqResponsePaymentList,
        last_runmoment: datetime = None,
    ) -> bool:
        """
        Check if should load more payments:
        - If there is no previous page, return False
        - If the earliest payment is older than the last runmoment, return False

        Args:
            payment_list_response: The response from the bunq API
            last_runmoment: The last runmoment, used to determine if we should load more payments.
                If not provided, all payments will be loaded.
        """
        if not payment_list_response.pagination.has_previous_page():
            return False
        if not last_runmoment:
            return True
        earliest_payment = payment_list_response.value[-1]
        earliest_payment_date = parse(earliest_payment.created)
        return earliest_payment_date > last_runmoment

    def get_payments_for_account(
        self, account: MonetaryAccountBank, last_runmoment: datetime = None
    ) -> List[Payment]:
        """
        Get the payments of an account.

        Args:
            account: The account to get the payments for
            last_runmoment: The last runmoment, used to determine if we should load more payments.
                If not provided, all payments will be loaded.
        """
        account_id = account.id_
        payments = []
        page_count = self.PAYMENTS_PER_PAGE
        pagination = Pagination()
        pagination.count = page_count

        # For first query, only param is the count param
        params = pagination.url_params_count_only
        # Loop over pages
        while True:
            query_result = endpoint.Payment.list(
                monetary_account_id=account_id, params=params
            )
            # Convert to dict
            current_payments = query_result.value
            payments.extend(current_payments)
            if self._should_continue_loading_payments(query_result, last_runmoment):
                # Use previous_page since ordering is new to old
                params = query_result.pagination.url_params_previous_page
            else:
                break
        self.logger.info(f"Loaded {len(payments)} payments for account {account_id}")
        return payments

    def get_accounts(self) -> List[MonetaryAccountBank]:
        """
        Get a list of all Bunq accounts
        """
        try:
            response = endpoint.MonetaryAccount.list().value
            accounts = [a.get_referenced_object() for a in response]
            self.logger.info(f"Loaded {len(accounts)} bunq accounts")
            return accounts
        except Exception as e:
            self.logger.error(f"Could not load bunq accounts: {e}")
            raise Exception(f"Could not load bunq accounts: {e}")
