"""Define the configuration variables for the project.

Todo: Store in json, read to class.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
CONFIG_DIR = PROJECT_DIR / ".." / "config"

BUNQ_CONFIG_DIR = CONFIG_DIR / "bunq"
BUNQ_CALLBACK_INDEX = "bunq_callback"
BUNQ_ONETIME_API_TOKEN_INDEX = "bunq_api_token"  # noqa: S105
BUNQ_CONFIG_INDEX = "bunq_config"

METADATA_DIR = PROJECT_DIR / "metadata"
LOGS_DIR = PROJECT_DIR / ".." / "logs"
CACHE_DIR = PROJECT_DIR / ".." / "cache"
LOGS_FILE = LOGS_DIR / "logs.log"
MLSERVER_CONFIG_DIR = CONFIG_DIR / "mlserver/models"
