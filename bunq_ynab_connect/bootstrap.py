"""
Initialize the dependency injection container and inject dependencies.
"""
from dotenv import find_dotenv, load_dotenv


def _load_env():
    load_dotenv(find_dotenv())


def bootstrap_di():
    """
    Inject dependencies into the dependency injection container.
    """
    _load_env()
