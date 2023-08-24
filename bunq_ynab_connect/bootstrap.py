"""
Initialize the dependency injection container and inject dependencies.
"""
import logging
import os

from dotenv import find_dotenv, load_dotenv
from kink import di
from pymongo import MongoClient
from pymongo.database import Database

from bunq_ynab_connect.helpers.config import LOGS_DIR, LOGS_FILE


def _load_env():
    load_dotenv(find_dotenv())


def _get_logger(name: str):
    """
    Get a logger with a file handler.
    """
    logger = logging.getLogger(name)
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    os.makedirs(LOGS_DIR, exist_ok=True)
    fhandler = logging.FileHandler(filename=LOGS_FILE, mode="a")
    formatter = logging.Formatter(log_fmt)
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)
    return logger


def bootstrap_di():
    """
    Inject dependencies into the dependency injection container.
    """
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
