import json
from collections.abc import Callable
from enum import Enum
from logging import LoggerAdapter
from typing import Any

from kink import inject
from requests import HTTPError, Response, get, post

from bunq_ynab_connect.clients.bunq.session_activator import (
    SessionActivator,
)
from bunq_ynab_connect.clients.bunq.signer import Signer
from bunq_ynab_connect.helpers.json_dict import JsonDict


class BunqEnvironment(Enum):  # noqa: D101
    SANDBOX = "SANDBOX"
    PRODUCTION = "PRODUCTION"


@inject
class BaseClient:
    """Bunq base client. Exposes get and post requests to the bunq API.

    Attributes
    ----------
        environment (BunqEnvironment): The bunq environment to use.
            SANDBOX or PRODUCTION.
        session_activator (SessionActivator): Ensures a valid session is active.
        signer (Signer): Sings and verifies requests.
        bunq_config (JsonDict): Bunq config filem, stored as json.
        logger (LoggerAdapter): The logger

    """

    environment: BunqEnvironment
    session_activator: SessionActivator
    signer: Signer
    bunq_config: JsonDict
    logger: LoggerAdapter

    @inject
    def __init__(
        self,
        environment: BunqEnvironment,
        signer: Signer,
        bunq_config: JsonDict,
        logger: LoggerAdapter,
    ) -> None:
        self.environment = environment
        self.session_activator = SessionActivator(bunq_client=self, logger=logger)
        self.signer = signer
        self.bunq_config = bunq_config
        self.logger = logger

    def post(
        self,
        *,
        endpoint: str,
        data: dict | None = None,
        headers: dict | None = None,
        **endpoint_variables: Any,  # noqa: ANN401
    ) -> dict:
        """Make a POST request to the bunq API.

        Parameters
        ----------
        endpoint : str
            The endpoint to call.
        data : dict, optional
            The data to POST, by default None
        headers : dict, optional
            The headers to add to endpoint, by default None. Default headers are always
            added.
        endpoint_variables : Any
            The variables to format the endpoint url with.

        """
        headers = headers or {}
        data = data or {}
        headers = {
            **self._default_headers(endpoint),
            "X-Bunq-Client-Signature": self.signer.sign(data),
            **headers,
        }
        response = post(
            self._format_endpoint(endpoint, **endpoint_variables),
            data=json.dumps(data).encode(),
            headers=headers,
            timeout=5,
        )
        return self._handle_response(response)

    def get(
        self,
        *,
        endpoint: str,
        params: dict | None = None,
        **endpoint_variables: Any,  # noqa: ANN401
    ) -> dict:
        """Make a GET request to the bunq API.

        Parameters
        ----------
        endpoint : str
            The endpoint to call.
        params : dict, optional
            The params to inlcude in the url.
            added.
        endpoint_variables : Any
            The variables to format the endpoint url with.

        """
        headers = self._default_headers(endpoint)
        response = get(
            self._format_endpoint(endpoint, **endpoint_variables),
            params=params or {},
            headers=headers,
            timeout=5,
        )
        return self._handle_response(response)

    def get_paginated(
        self,
        *,
        endpoint: str,
        params: dict | None = None,
        continue_loading_pages: Callable[[list], bool] | None = None,
        page_size: int = 200,
        **endpoint_variables: Any,  # noqa: ANN401
    ) -> list:
        """Perform a GET request in a paginated way.

        List endpoints support pagination. Add the count paramter to the request,
        and continue loading pages until done. Assumption is we load newer pages first.
        Therefor done is when there is no older url, and the provided callback returns
        False.

        Parameters
        ----------
        endpoint : str
            The endpoint to call.
        params : dict, optional
            The parameters to pass to the endpoint, by default None
        continue_loading_pages : Callable[[list], bool], optional
            A callback to determine if more pages should be loaded, by default None.
            Should consume the last page, and return True if more pages should be loaded
        page_size : int, optional
            The amount of items to load per page, by default 200
        endpoint_variables : Any
            The variables to format the endpoint with.

        """
        params = params or {}
        done = False
        result = []
        while not done:
            response = self.get(
                endpoint=endpoint,
                data={**params, "count": page_size},
                **endpoint_variables,
            )
            last_page = response["Response"]
            result.extend(last_page)
            pagination = response["Pagination"]
            done = (
                "Pagination" not in response
                or "older_url" not in pagination
                or not pagination["older_url"]
            ) or (
                continue_loading_pages is not None
                and not continue_loading_pages(last_page)
            )
            if not done:
                endpoint = pagination["older_url"]
                page_size = None
        return result

    def _default_headers(self, endpoint: str) -> dict:
        """All required headers. Authentication is based on the endpoint."""
        return {
            "User-Agent": "bunqynab",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "X-Bunq-Client-Authentication": self._authentication_token(endpoint),
        }

    def _authentication_token(self, endpoint: str) -> str:
        """Decide the value for the Authentication header, based on the endpoint.

        If we need to use a session token, we ensure the session is active.
        This means we could recreate the session if needed.
        """
        if endpoint == "installation":
            return None
        if endpoint in ["device-server", "session-server"]:
            return self.session_activator.installation_token
        self.session_activator.ensure_session_active()
        return self.session_activator.session_token

    def _format_endpoint(self, endpoint: str, **params: Any) -> str:  # noqa: ANN401
        url = endpoint.format(**params)
        if url.startswith("/v1"):
            url = url.replace("/v1", "")
        if not url.startswith(self._base_url):
            url = f"{self._base_url}/{url}"
        return url

    def _handle_response(self, response: Response) -> dict:
        """Handle a response, by raising if error, else verifying the signature."""
        try:
            response.raise_for_status()
        except HTTPError:
            msg = f"Error in API request: {response.text}"
            self.logger.exception(msg)
            raise
        else:
            self.signer.verify(
                response.text, response.headers["X-Bunq-Server-Signature"]
            )
            return response.json()

    @property
    def _base_url(self) -> str:
        """Decide the url for the bunq environment.

        See [here](https://beta.doc.bunq.com/basics/moving-to-production)
        """
        if self.environment == BunqEnvironment.SANDBOX:
            return "https://public-api.SANDBOX.bunq.com/v1"
        if self.environment == BunqEnvironment.PRODUCTION:
            return "https://api.bunq.com/v1"
        msg = f"Invalid bunq environment: {self.environment}"
        raise ValueError(msg)
