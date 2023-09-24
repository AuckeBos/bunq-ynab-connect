import pickle
import shelve
from datetime import date, datetime
from functools import wraps
from logging import LoggerAdapter
from tempfile import TemporaryDirectory
from time import time
from typing import Any

import pytz
import requests
from kink import inject

from bunq_ynab_connect.helpers.config import CACHE_DIR
from mlflow import log_artifact


def now():
    return datetime.now(tz=pytz.timezone("Europe/Amsterdam"))


def get_public_ip():
    """
    Get the current public ip address
    """
    return requests.get("http://ipinfo.io/json").json()["ip"]


@inject
def cache(logger: LoggerAdapter, ttl: int = None):
    """
    Cache decorator, to cache the result of a function for some seconds
    :param ttl: Time to live of cache in seconds
    args
    """
    expires_at = time() + ttl

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args[1:]) + str(tuple(sorted(kwargs.items())))[1:-1]
            c = shelve.open(f"{CACHE_DIR}/{func.__name__}_{key}")
            is_expired = (
                "expires_at" in c
                and c["expires_at"] is not None
                and c["expires_at"] < time()
            )
            cache_valid = not is_expired and "value" in c
            if not cache_valid:
                value = func(*args, **kwargs)
                c["value"] = value
                c["expires_at"] = expires_at
            else:
                logger.info(f"Using cached value for {func.__name__}_{key}")
                value = c["value"]
            c.close()
            return value

        return wrapper

    return decorator


def date_to_datetime(_date: date) -> datetime:
    return datetime(_date.year, _date.month, _date.day)


def object_to_mlflow(obj: Any, name: str) -> None:
    """
    Save an object to an artifact by:
    - Saving the object to a temp pickle file
    - Saving the temp pickle file as artifact in the current mlfow run
    Parameters
    ----------
    obj: Any
        The dict to save
    name: str
        The artefact name
    """
    with TemporaryDirectory() as temp_dir:
        path = f"{temp_dir}/{name}"
        with open(path, "wb") as tmp_file:
            pickle.dump(obj, tmp_file)
        log_artifact(path)
        return path
