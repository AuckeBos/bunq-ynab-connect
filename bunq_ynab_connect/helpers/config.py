"""Define the configuration variables for the project.

Todo: Store in json, read to class.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).parents[1]
CONFIG_DIR = PROJECT_DIR / ".." / "config"
BUNQ_CONFIG_FILE = CONFIG_DIR / "bunq.cfg"
METADATA_DIR = PROJECT_DIR / "metadata"
LOGS_DIR = PROJECT_DIR / ".." / "logs"
CACHE_DIR = PROJECT_DIR / ".." / "cache"
LOGS_FILE = LOGS_DIR / "logs.log"
MLSERVER_CONFIG_DIR = CONFIG_DIR / "mlserver/models"
