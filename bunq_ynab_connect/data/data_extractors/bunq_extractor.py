import json
import os
from logging import LoggerAdapter
from typing import List

from bunq import ApiEnvironmentType, Pagination
from bunq.sdk.context.api_context import ApiContext
from bunq.sdk.context.bunq_context import BunqContext
from bunq.sdk.model.generated import endpoint
from bunq.sdk.model.generated.endpoint import Payment
from kink import inject

from bunq_ynab_connect.data.data_extractors.abstract_extractor import AbstractExtractor
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.config import BUNQ_CONFIG_FILE
from bunq_ynab_connect.helpers.general import cache, get_public_ip


class BunqExtractor(AbstractExtractor):
    @inject
    def __init__(self, storage: AbstractStorage, logger: LoggerAdapter) -> None:
        super().__init__("bunq_transactions", storage, logger)
        self._load_api_context()

    def _load_api_context(self):
        """
        Initialize context, ran on init
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

    @cache(ttl=60 * 60 * 24)
    def load_for_account(self, account_id: int) -> List:
        """
        Get the payments of an account
        """
        # Max allowed count is 200
        payments = []
        page_count = 200
        pagination = Pagination()
        pagination.count = page_count

        # For first query, only param is the count param
        params = pagination.url_params_count_only
        should_continue = True
        # Loop over pages
        while should_continue:
            query_result = endpoint.Payment.list(
                monetary_account_id=account_id, params=params
            )
            # Convert to dict
            current_payments = [json.loads(pay.to_json()) for pay in query_result.value]
            payments.extend(current_payments)
            should_continue = query_result.pagination.has_previous_page()
            if should_continue:
                # Use previous_page since ordering is new to old
                params = query_result.pagination.url_params_previous_page
        self.logger.info(f"Loaded {len(payments)} payments for account {account_id}")
        return payments

    @cache(ttl=60 * 60 * 24)
    def get_bunq_account_ids(self) -> List:
        """
        Get a list of all bunq account ids
        """
        accounts = endpoint.MonetaryAccount.list().value
        ids = [a.get_referenced_object().id_ for a in accounts]
        self.logger.info(f"Loaded {len(ids)} bunq accounts")
        return ids

    def load(self) -> List:
        """
        Load the data from the source.
        Loads all payments from all accounts
        """
        account_ids = self.get_bunq_account_ids()
        payments = []
        for id in account_ids:
            payments.extend(self.load_for_account(id))
        return payments
