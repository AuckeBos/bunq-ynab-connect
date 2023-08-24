"""
Define the configuration variables for the project.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".."
CONFIG_DIR = PROJECT_DIR / "config"
BUNQ_CONFIG_FILE = f"{CONFIG_DIR}/bunq.cfg"
METADATA_DIR = CONFIG_DIR / "metadata"
LOGS_DIR = PROJECT_DIR / ".." / "logs"
CACHE_DIR = PROJECT_DIR / ".." / "cache"
LOGS_FILE = LOGS_DIR / "logs.log"