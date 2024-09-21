from datetime import datetime
from functools import partial
from logging import LoggerAdapter

import pytz
from dateutil.parser import parse
from kink import inject
from pydantic import BaseModel

from bunq_ynab_connect.clients.bunq.base_client import BaseClient
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.json_dict import JsonDict
from bunq_ynab_connect.models.bunq_account import (
    BunqAccount,
)


class Callback(BaseModel):
    """Pydantic model for a callback."""

    notification_target: str
    category: str

    @staticmethod
    def from_api_response(response: dict) -> "Callback":
        return Callback(
            notification_target=response["NotificationFilterUrl"][
                "notification_target"
            ],
            category=response["NotificationFilterUrl"]["category"],
        )


class BunqClient:
    """Client to expose specific bunq API calls.

    Uses the BaseClient to make requests.

    Attributes
    ----------
        storage (AbstractStorage): The storage.
        logger (LoggerAdapter): The logger.
        base_client (BaseClient): The base client, used to make requests.
        bunq_config (JsonDict): The bunq config file, stored as json.
        ITEMS_PER_PAGE (int): The amount of items to load per page
            for paginated requests.

    """

    storage: AbstractStorage
    logger: LoggerAdapter
    base_client: BaseClient
    bunq_config: JsonDict
    ITEMS_PER_PAGE: int = 100

    @inject
    def __init__(
        self,
        storage: AbstractStorage,
        logger: LoggerAdapter,
        base_client: BaseClient,
        bunq_config: JsonDict,
    ) -> None:
        self.storage = storage
        self.logger = logger
        self.base_client = base_client
        self.bunq_config = bunq_config

    def _should_continue_loading_payments(
        self,
        last_page: list,
        last_runmoment: datetime | None = None,
    ) -> bool:
        """Check if should load more payments.

        If the oldest payment of the last page is newer than the last runmoment,
        we should continue loading payments older payments.
        """
        if not last_runmoment:
            return True
        earliest_payment = last_page[-1]
        created_at = parse(earliest_payment["Payment"]["created"]).replace(
            tzinfo=pytz.UTC
        )
        return created_at > last_runmoment

    def get_payments_for_account(
        self, account: BunqAccount, last_runmoment: datetime | None = None
    ) -> list[dict]:
        """Get the payments of an account.

        If the last runmoment is provided, only payments after that moment are loaded.
        """
        user_id = self.user_id
        payments: list[dict] = self.base_client.get_paginated(
            endpoint="user/{user_id}/monetary-account/{account_id}/payment",
            user_id=user_id,
            account_id=account.id,
            page_size=self.ITEMS_PER_PAGE,
            continue_loading_pages=partial(
                self._should_continue_loading_payments, last_runmoment=last_runmoment
            ),
        )
        # Remove payments after last runmoment, and flatten
        if last_runmoment:
            payments = [
                p["Payment"]
                for p in payments
                if parse(p["Payment"]["created"]).replace(tzinfo=pytz.UTC)
                > last_runmoment
            ]
        if len(payments) > 0:
            self.logger.info(
                "Loaded %s payments for account %s", len(payments), account.id
            )
        return payments

    def get_accounts(
        self,
    ) -> list[BunqAccount]:
        """Get all bunq accounts for the user."""
        try:
            accounts_response: list[dict] = self.base_client.get_paginated(
                endpoint="user/{user_id}/monetary-account",
                user_id=self.user_id,
                page_size=self.ITEMS_PER_PAGE,
            )
            accounts: list[BunqAccount] = [
                BunqAccount.from_dict(v)
                for account in accounts_response
                for v in account.values()
            ]
            self.logger.info("Loaded %s bunq accounts", len(accounts))
        except Exception as e:
            msg = "Could not load bunq accounts"
            self.logger.exception(msg)
            raise ValueError(msg) from e
        else:
            return accounts

    def exchange_pat(self, pat: str) -> None:
        """Trade a PAT for a new bunq config file.

        Remove some config variables, to enforce an exchange.
        """
        old_config = self.bunq_config.data
        config = old_config.copy()
        config["api_token"] = pat
        del config["installation_context"]
        del config["session_context"]
        self.bunq_config.save(config)
        try:
            self.base_client.session_activator.ensure_session_active()
        except Exception as e:
            self.bunq_config.save(old_config)
            msg = "Could not trade PAT for new bunq config"
            self.logger.exception(msg)
            raise ValueError(msg) from e

        self.logger.warning(
            "Token traded. Make sure to remove the old token in the app"
        )

    def add_callback(self, url: str) -> None:
        """Add a callback to the bunq API."""
        if self._callback_exists(url):
            return
        self.logger.info("Adding callback for %s", url)
        callbacks = self._get_callbacks()
        callbacks.append(Callback(notification_target=url, category="MUTATION"))
        self._set_callbacks(callbacks)

    def _set_callbacks(self, callbacks: list[Callback]) -> None:
        self.base_client.post(
            endpoint="/user/{user_id}/notification-filter-url",
            user_id=self.user_id,
            data={"notification_filters": [c.dict() for c in callbacks]},
        )

    def _callback_exists(self, url: str) -> bool:
        return any(c.notification_target == url for c in self._get_callbacks())

    def _get_callbacks(self) -> list[Callback]:
        return [
            Callback(
                notification_target=c["NotificationFilterUrl"]["notification_target"],
                category=c["NotificationFilterUrl"]["category"],
            )
            for c in self.base_client.get(
                endpoint="/user/{user_id}/notification-filter-url",
                user_id=self.user_id,
            )["Response"]
        ]

    def remove_callback(self, url: str) -> None:
        if not self._callback_exists(url):
            return
        self.logger.info("Removing callback for %s", url)
        callbacks = self._get_callbacks()
        callbacks = [c for c in callbacks if c.notification_target != url]
        self._set_callbacks(callbacks)

    @property
    def user_id(self) -> int:
        if not self.bunq_config["session_context.user_person_id.id"]:
            self.base_client.session_activator.ensure_session_active()
        return self.bunq_config["session_context.user_person_id.id"]
