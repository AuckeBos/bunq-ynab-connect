import json
from logging import LoggerAdapter
from pathlib import Path

from kink import inject


class JsonDict(dict):
    """Helper class to real-time read and write a json file."""

    logger: LoggerAdapter

    @inject
    def __init__(self, logger: LoggerAdapter, path: Path) -> None:
        self.logger = logger
        self.path = path

    @property
    def data(self) -> dict:
        if not self.path.exists():
            return {}
        with self.path.open("r") as f:
            return json.load(f)

    def __getitem__(self, key: str) -> str:
        """Get a value from the config file.

        Parameters
        ----------
        key : str
            The key to get the value for. Can contain dots

        """
        config = self.data
        for part in key.split("."):
            config = config.get(part, None)
            if config is None:
                return None
        return config

    def __setitem__(self, key: str, value: str) -> None:
        """Set a value in the config file."""
        self.update({key: value})

    def update(self, data: dict) -> None:
        self.save(_merge_dicts(self.data, data))

    def save(self, config: dict) -> None:
        with self.path.open("w") as f:
            json.dump(config, f, indent=4)


def _merge_dicts(original: dict, new: dict) -> dict:
    """Recursively merge two dictionaries."""
    for key, value in new.items():
        if (
            key in original
            and isinstance(original[key], dict)
            and isinstance(value, dict)
        ):
            original[key] = _merge_dicts(original[key], value)
        else:
            original[key] = value
    return original
