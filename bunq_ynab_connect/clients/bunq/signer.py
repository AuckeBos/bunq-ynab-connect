import json
from base64 import b64decode, b64encode
from logging import LoggerAdapter

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey
from cryptography.hazmat.primitives.hashes import SHA256
from kink import inject

from bunq_ynab_connect.helpers.json_dict import JsonDict


@inject
class Signer:
    """Signs and verifies Bunq API requests.

    Loads its keys from the bunq config file.

    Attributes
    ----------
        logger (LoggerAdapter): The logger
        bunq_config (JsonDict): The bunq config file, stored as json.
            All required info is stored here, by the SessionActivator.

    """

    logger: LoggerAdapter
    bunq_config: JsonDict

    @inject
    def __init__(
        self,
        logger: LoggerAdapter,
        bunq_config: JsonDict,
    ) -> None:
        self.logger = logger
        self.bunq_config = bunq_config

    def sign(self, data: dict) -> str:
        return b64encode(
            self._private_key.sign(json.dumps(data).encode(), PKCS1v15(), SHA256())
        )

    def verify(self, data: str, signature: str) -> bool:
        if not self._server_public_key:
            # Only allowed during API activation.
            self.logger.warning(
                "No server public key found; skipping signature verification;"
            )
            return True
        try:
            self._server_public_key.verify(
                b64decode(signature),
                data.encode(),
                PKCS1v15(),
                SHA256(),
            )
        except InvalidSignature:
            self.logger.exception("Invalid response signature;")
            raise
        else:
            return True

    @property
    def _private_key(self) -> RSAPrivateKey:
        return serialization.load_pem_private_key(
            self.bunq_config["installation_context.private_key_client"].encode(),
            password=None,
        )

    @property
    def _server_public_key(self) -> RSAPublicKey | None:
        if key := self.bunq_config["installation_context.public_key_server"]:
            return serialization.load_pem_public_key(key.encode())
        return None
