"""Initialize the dependency injection container and inject dependencies."""

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from kink import di, inject
from prefect.exceptions import MissingContextError
from prefect.logging import get_logger, get_run_logger
from pymongo import MongoClient
from pymongo.database import Database
from ynab.models.account import Account

from bunq_ynab_connect.clients.bunq.base_client import BunqEnvironment
from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.data.storage.mongo_storage import MongoStorage
from bunq_ynab_connect.helpers.config import (
    BUNQ_CALLBACK_INDEX,
    BUNQ_CONFIG_DIR,
    BUNQ_CONFIG_INDEX,
    BUNQ_ONETIME_API_TOKEN_INDEX,
    CACHE_DIR,
    CONFIG_DIR,
    LOGS_DIR,
    MLSERVER_CONFIG_DIR,
    MLSERVER_PREDICTION_URL_INDEX,
    MLSERVER_REPOSITORY_URL_INDEX,
)
from bunq_ynab_connect.helpers.json_dict import JsonDict


def _load_env() -> None:
    load_dotenv(find_dotenv())


def _get_logger(name: str) -> logging.LoggerAdapter:
    """Get the logger.

    If we can get the prefect logger (we are running in a prefect flow), use it
    If not, create a new logger.
    """
    try:
        logger = get_run_logger()
    except MissingContextError:
        logger = get_logger(name)
    logger.setLevel(logging.DEBUG)
    return logger


def bootstrap_di() -> None:
    """Inject dependencies into the dependency injection container."""
    for dir_ in [LOGS_DIR, CACHE_DIR, CONFIG_DIR, MLSERVER_CONFIG_DIR, BUNQ_CONFIG_DIR]:
        Path.mkdir(dir_, exist_ok=True, parents=True)
    # Env
    _load_env()

    # Logging
    # Use factory, to retry getting the prefect logger each time
    di.factories[logging.LoggerAdapter] = lambda _: _get_logger("logger")

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
    # Bunq config
    di[BunqClient] = lambda _: BunqClient()
    di[BUNQ_CALLBACK_INDEX] = os.getenv("BUNQ_CALLBACK_HOST")
    di[BUNQ_ONETIME_API_TOKEN_INDEX] = os.getenv("BUNQ_ONETIME_TOKEN")
    bunq_environment = BunqEnvironment(os.getenv("BUNQ_ENVIRONMENT", "SANDBOX"))
    di[BunqEnvironment] = bunq_environment
    di[BUNQ_CONFIG_INDEX] = JsonDict(
        path=Path(BUNQ_CONFIG_DIR / f"bunq_{bunq_environment.name}.cfg")
    )
    # Model serving config
    di[MLSERVER_PREDICTION_URL_INDEX] = (
        "{server_url}/v2/models/{{budget_id}}/infer".format(
            server_url=os.getenv("MLSERVER_URL")
        )
    )
    di[MLSERVER_REPOSITORY_URL_INDEX] = (
        "{server_url}/v2/repository/models/{{budget_id}}".format(
            server_url=os.getenv("MLSERVER_URL")
        )
    )


def monkey_patch_ynab() -> None:
    """Override bugs in Ynab classes.

    Some classes have a bugged attributes.
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

        def fixed_setter(self, value):  # noqa: ANN001, ANN202
            setattr(self, f"_{attribute}", value)  # noqa: B023

        setattr(Account, attribute, fixed_setter)


@inject
def import_mlserver_windows_friendly(logger: logging.LoggerAdapter) -> None:
    """MLServer has issues on windows. It seems a reinstall solves the issue.

    https://github.com/SeldonIO/MLServer/issues/1022
    """
    try:
        import mlserver  # noqa: F401
    except Exception:  # noqa: BLE001
        import subprocess

        logger.warning("Bad mlserver import; trying to install mlserver")
        install_cmd = "pip install mlserver"
        subprocess.Popen(install_cmd, shell=True, stdout=subprocess.DEVNULL)  # noqa: S602
        logger.debug("Good mlserver reinstall;")
    finally:
        from mlserver.codecs import PandasCodec  # noqa: F401

        logger.debug("Good mlserver import;")


def monkey_patch_mlserver() -> None:
    """Fix an issue with mlserver, where datetime fiels cannot be encoded."""
    from mlserver.codecs.numpy import _NumpyToDatatype

    _NumpyToDatatype["M"] = "BYTES"
