from collections.abc import Generator
from contextlib import contextmanager
from logging import LoggerAdapter

from kink import inject

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.helpers.general import now


@inject
class PaymentQueue:
    """Class to handle the queue of payments to be synced.

    Attributes
    ----------
        logger: The logger to use to log messages.
        storage: The storage class to use to store the queue.

    Usage:
        while queue := PaymentQueue(logger, storage):
            with queue.pop() as payment_id:
                # Process payment

    """

    TABLE_NAME = "payment_queue"
    logger: LoggerAdapter
    storage: AbstractStorage

    @inject
    def __init__(self, logger: LoggerAdapter, storage: AbstractStorage):
        self.logger = logger
        self.storage = storage

    def get_payment_id(self) -> str:
        """Get the payment id of the first non-synced payment in the queue.

        Order is determined by the updated_at column (first in, first out)
        """
        payment = self.storage.find_one(
            self.TABLE_NAME, {("synced_at", "eq", None)}, "updated_at"
        )
        if payment is None:
            msg = "No payments in queue"
            raise IndexError(msg)
        return payment["payment_id"]

    def mark_as_synced(self, payment_id: str) -> None:
        """Mark a payment as synced."""
        data = {
            "payment_id": payment_id,
            "synced_at": now(),
        }
        self.storage.upsert(self.TABLE_NAME, [data])

    def __bool__(self) -> bool:
        """To support the usage of the queue in a while loop."""
        return self.storage.count(self.TABLE_NAME, {("synced_at", "eq", None)}) > 0

    def add(self, payment_id: str) -> None:
        """Add a payment to the queue (if it doesn't already exist)."""
        data = {
            "payment_id": payment_id,
            "synced_at": None,
        }
        self.storage.insert_if_not_exists(self.TABLE_NAME, [data])

    @contextmanager
    def pop(self) -> Generator[str, None, None]:
        """Pop a payment from the queue.

        The payment is removed from the queue when the context is exited cleanly.
        Can be used in a with statement.
        """
        payment_id = self.get_payment_id()
        try:
            yield payment_id
            self.mark_as_synced(payment_id)
        except Exception:
            self.logger.exception("Error processing payment %s", payment_id)
            raise
