import shelve
from datetime import datetime
from functools import wraps
from logging import LoggerAdapter
from time import time

import pytz
import requests
from kink import inject

from bunq_ynab_connect.helpers.config import CACHE_DIR


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
