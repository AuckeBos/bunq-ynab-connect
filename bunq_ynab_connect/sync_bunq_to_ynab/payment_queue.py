from contextlib import contextmanager
from logging import LoggerAdapter
from typing import Generator

from kink import inject

from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage


@inject
class PaymentQueue:
    """
    Class to handle the queue of payments to be synced.

    Attributes:
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
        """
        Get the payment id of the first payment in the queue.
        Order is determined by the updated_at column (first in, first out)
        """
        payment = self.storage.find_one(self.TABLE_NAME, {}, "updated_at")
        if payment is None:
            raise IndexError("No payments in queue")
        return payment["payment_id"]

    def remove_payment(self, payment_id: str):
        """
        Remove a payment from the queue. Called when the payment has been processed.
        """
        self.storage.delete(self.TABLE_NAME, {"payment_id": payment_id})

    def __bool__(self):
        """
        To support the usage of the queue in a while loop.
        """
        return self.storage.count(self.TABLE_NAME) > 0

    def add(self, payment_id: str):
        """
        Add a payment to the queue.
        """
        self.storage.upsert(self.TABLE_NAME, [{"payment_id": payment_id}])

    @contextmanager
    def pop(self) -> Generator[str, None, None]:
        """
        Pop a payment from the queue.
        The payment is removed from the queue when the context is exited cleanly.
        Can be used in a with statement.
        """
        payment_id = self.get_payment_id()
        try:
            yield payment_id
            self.remove_payment(payment_id)
        except Exception as e:
            self.logger.error(f"Error processing payment {payment_id}: {e}")
            raise e
