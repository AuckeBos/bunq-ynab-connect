import platform
from datetime import datetime, timedelta
from logging import LoggerAdapter
from typing import TYPE_CHECKING

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from kink import inject
from requests import HTTPError

from bunq_ynab_connect.helpers.general import now
from bunq_ynab_connect.helpers.json_dict import JsonDict

if TYPE_CHECKING:
    from bunq_ynab_connect.clients.bunq.base_client import BaseClient


class SessionActivator:
    """Handles session initialization and activation.

    Uses a json file to store all tokens and info.
    Folows the flow as described [here](https://beta.doc.bunq.com/basics/authentication)

    Applies above flow back-to-forth. This means that when it calls an endpoint:
    - Check if session should be created, if so
    - Check if API token should be activated, if so
    - Check if RSA key should be activated, if so
    - Check if RSA key should be generated, if so
    - Generate RSA key

    This way, we do not need to call an initial setup() method, but can call
    any endpoint directly after initializing the class.

    Attributes
    ----------
        logger (LoggerAdapter): The logger
        api_token_from_env (str): The API token, read from env vars. Used for the very
            first setup. Token is stored in json config once activated.
        bunq_client (BaseClient): The bunq client
        bunq_config (JsonDict): The bunq config file, stored as json.
            All required info is stored here, including session tokens.
        SAFETY_MARGIN_SESSION_EXPIRATION_SECONDS (int): The safety margin for session
        expiration. If the session expires in less than this amount of seconds, a new
        session is created.

    """

    logger: LoggerAdapter
    api_token_from_env: str
    bunq_client: "BaseClient"
    bunq_config: JsonDict
    SAFETY_MARGIN_SESSION_EXPIRATION_SECONDS = 60

    @inject
    def __init__(
        self,
        bunq_api_token: str,
        bunq_client: "BaseClient",
        bunq_config: JsonDict,
        logger: LoggerAdapter,
    ) -> None:
        self.api_token_from_env = bunq_api_token
        self.bunq_client = bunq_client
        self.bunq_config = bunq_config
        self.logger = logger

    def ensure_session_active(self) -> None:
        if self.session_token is None:
            self.logger.info("Session expired;")
            self._create_session()

    @property
    def installation_token(self) -> str | None:
        return self.bunq_config["installation_context.token"]

    @property
    def session_token(self) -> str | None:
        """Return session token from config, if not expired."""
        expires_at = self.bunq_config["session_context.expires_at"]
        if expires_at is None or datetime.fromisoformat(expires_at) < now():
            return None
        return self.bunq_config["session_context.token"]

    def _create_session(self) -> None:
        """(Re)create a session. Activate API token if not done yet.

        Third and last step in the bunq authentication flow.
        Stores the session token and expiration in the config.
        """
        if self.session_token is not None:
            return
        if not self._api_token_activated:
            self._activate_api_token()

        self.logger.info("Creating new API session;")

        response = self.bunq_client.post(
            endpoint="session-server",
            data={
                "secret": self.api_token,
            },
        )
        try:
            self.bunq_config.update(
                {
                    "session_context": {
                        "token": response["Response"][1]["Token"]["token"],
                        "expires_at": self._compute_expiration(
                            response["Response"][2]["UserPerson"]["session_timeout"]
                        ).isoformat(),
                        "user_person_id": response["Response"][2]["UserPerson"],
                    }
                }
            )
        except (KeyError, IndexError) as e:
            msg = f"Could not create API session; response {response};"
            self.logger.exception(msg)
            raise ValueError(msg) from e

    @property
    def _api_token_activated(self) -> bool:
        """Check if the token activation values are present in the config."""
        return (
            self.bunq_config["api_token"] is not None
            and self.bunq_config["installation_context.device_id"] is not None
        )

    def _activate_api_token(self) -> None:
        """Activate the API token. Generate RSA key if not done yet.

        Activate RSA key if not done yet.

        Second step in the bunq authentication flow.
        Stores the API token and device id in the config.
        """
        if self._api_token_activated:
            return
        if not self._rsa_key_activated:
            self._activate_rsa_key()
        self.logger.info("Activating Onetime API token")
        try:
            response = self.bunq_client.post(
                endpoint="device-server",
                data={
                    "description": f"bunqynab_{platform.node()}",
                    "secret": self.api_token,
                },
            )
            device_id = response["Response"][0]["Id"]["id"]
        except HTTPError as e:
            # If device was activated before, continue.
            if (
                e.response.status_code == 400  # noqa: PLR2004
                and "device already exists" in e.response.text
            ):
                # Save to prevent retrying
                device_id = "device already exists"
            else:
                raise
        except (KeyError, IndexError) as e:
            msg = f"Could not activate onteimte API token; response {response};"
            self.logger.exception(msg)
            raise ValueError(msg) from e
        self.bunq_config.update(
            {
                "installation_context": {
                    "device_id": device_id,
                },
                "api_token": self.api_token,
            }
        )

    @property
    def _rsa_key_activated(self) -> bool:
        """Check if the RSA key activation values are present in the config."""
        return self.installation_token is not None

    def _activate_rsa_key(self) -> None:
        """Activate the RSA key. Generate RSA key if not done yet.

        First step in the bunq authentication flow.
        Stores the server RSA key and installation token in the config.
        Server RSA key is used to verify the server responses.
        """
        if self._rsa_key_activated:
            return
        if not self._rsa_key_generated:
            self._generate_rsa_key()
        self.logger.info("Generating installation token")
        response = self.bunq_client.post(
            endpoint="installation",
            data={
                "client_public_key": self.bunq_config[
                    "installation_context.public_key_client"
                ]
            },
        )
        try:
            self.bunq_config.update(
                {
                    "installation_context": {
                        "public_key_server": response["Response"][2]["ServerPublicKey"][
                            "server_public_key"
                        ],
                        "token": response["Response"][1]["Token"]["token"],
                    }
                }
            )
        except (KeyError, IndexError) as e:
            msg = f"Could not activate RSA key; response {response};"
            self.logger.exception(msg)
            raise ValueError(msg) from e

    @property
    def _rsa_key_generated(self) -> bool:
        """Check if the RSA key is generated."""
        return self.bunq_config["installation_context.private_key_client"] is not None

    def _generate_rsa_key(self) -> None:
        """Use the cryptography library to generate an RSA key pair.

        Key is used to initialize device-server, and to sign the requests.
        """
        if self._rsa_key_generated:
            return

        self.logger.info("Generating RSA key")
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_key = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_key = key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.bunq_config.update(
            {
                "installation_context": {
                    "private_key_client": private_key.decode(),
                    "public_key_client": public_key.decode(),
                }
            }
        )

    def _compute_expiration(self, session_timeout: int) -> datetime:
        return (
            now()
            + timedelta(seconds=session_timeout)
            - timedelta(seconds=self.SAFETY_MARGIN_SESSION_EXPIRATION_SECONDS)
        )

    @property
    def api_token(self) -> str:
        """Once the token is activated, store it in config and use from there."""
        if token := self.bunq_config["api_token"]:
            return token
        return self.api_token_from_env
