from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from json import JSONDecodeError
from logging import LoggerAdapter

from fastapi import FastAPI, Request, Response
from kink import di, inject
from prefect.deployments import run_deployment
from pydantic.error_wrappers import ValidationError

from bunq_ynab_connect.clients.bunq_client import BunqClient
from bunq_ynab_connect.data.storage.abstract_storage import AbstractStorage
from bunq_ynab_connect.models.bunq_payment import BunqPayment


@inject
def ensure_callback_exists(bunq_callback: str, client: BunqClient) -> None:
    url = f"https://{bunq_callback}/payment"
    client.add_callback(url)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Ensure the callback url is registered in Bunq, before the server starts."""
    ensure_callback_exists()
    logger = di[LoggerAdapter]
    logger.info("Started callback server.")
    yield


@inject
async def process_payment(payment: BunqPayment, storage: AbstractStorage) -> None:
    """Add the payment to stroage, then trigger a sync_payment deployment run."""
    storage.upsert("bunq_payments", [payment.dict()])
    flow_run = await run_deployment(
        name="sync-payment/sync_payment",
        parameters={"payment_id": payment.id},
        timeout=0,
    )
    logger = di[LoggerAdapter]
    logger.info("Syncing payment %s in flow run %s", payment.id, flow_run.id)


app = FastAPI(lifespan=lifespan)


@app.post("/payment")
async def receive_payment(request: Request, response: Response) -> str:
    """Receive a payment, sync it to Ynab and return the result."""
    logger = di[LoggerAdapter]
    try:
        request_json = await request.json()
        payment_data = request_json["NotificationUrl"]["object"]["Payment"]
        payment = BunqPayment.parse_obj(payment_data)
        await process_payment(payment)
    except (IndexError, ValidationError, JSONDecodeError) as e:
        logger.exception("Invalid payment received: %s", await request.body())
        response.status_code = 400
        return f"Invalid payment: {e}"
    else:
        response.status_code = 200
        return "OK"
