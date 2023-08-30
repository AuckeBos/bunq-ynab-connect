"""
Initialize the dependency injection container and inject dependencies.
"""
import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv
from kink import di
from prefect import get_run_logger
from pymongo import MongoClient
from pymongo.database import Database
from ynab.models.account import Account

from bunq_ynab_connect.clients.ynab_client import YnabClient
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.data.storage.mongo_storage import MongoStorage
from bunq_ynab_connect.helpers.config import CACHE_DIR, CONFIG_DIR, LOGS_DIR, LOGS_FILE


def _load_env():
    load_dotenv(find_dotenv())


def _get_logger(name: str):
    """
    Get the logger.
    If we can get the prefect logger (we are running in a prefect flow), use it
    If not, create a new logger.
    """
    try:
        logger = get_run_logger()
    except:
        logger = logging.getLogger(name)
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        os.makedirs(LOGS_DIR, exist_ok=True)
        fhandler = logging.FileHandler(filename=LOGS_FILE, mode="a")
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(log_fmt)
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def bootstrap_di():
    """
    Inject dependencies into the dependency injection container.
    """
    for dir in [LOGS_DIR, CACHE_DIR, CONFIG_DIR]:
        os.makedirs(dir, exist_ok=True)
    # Env
    _load_env()

    # Logging
    di[logging.LoggerAdapter] = lambda di: _get_logger("logger")

    # MongoDB
    di[MongoClient] = lambda _di: MongoClient(
        os.getenv("MONGO_URI"),
        username=os.getenv("MONGO_USER"),
        password=os.getenv("MONGO_PASSWORD"),
        serverSelectionTimeoutMS=1000,
        connectTimeoutMS=1000,
    )
    di[Database] = lambda _di: _di[MongoClient][os.getenv("MONGO_DB", "MYDB")]

    # Set the MongoStorage as the default storage
    di[AbstractStorage] = lambda _di: MongoStorage()


def monkey_patch_ynab():
    """
    Some ynab classes have bugs. Override the functions with those bugs here,
    to prevent exceptions.

    Some classes have som bugged attributes.
    In an API response, they sometimes get a value which 'is not allowed'.
    In such cases, an exception would occur if the class is instantiated. To
    prevent this, we override the set() functions of those properties. The new
    function definition simply sets the value, skipping the 'raise exception if
    value is None' part.

    """

    bugged_attributes = ["type"]

    # Fix Account
    bugged_attributes = ["type"]
    for attribute in bugged_attributes:

        def fixed_setter(self, value):
            setattr(self, f"_{attribute}", value)

        setattr(Account, attribute, fixed_setter)
