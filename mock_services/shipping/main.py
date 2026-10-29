"""Mock shipping API (W1.9): `uvicorn mock_services.shipping.main:app --port 8002`.

Shipment status and courier scans for order-status complaints. Every endpoint except
`/health` needs the `X-API-Key` header (see `mock_services.common`). Reads can be made to
fail or to respond slowly, so the agent's "status not available right now" path
(shipping_delays_v1 section 5) can be tested.
"""

import time
from datetime import UTC, date, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Query, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mock_services.common import (
    FaultRequestBase,
    MockError,
    add_error_handler,
    add_fault_routes,
    api_key_check,
    raise_fault_error,
)
from mock_services.shipping.store import STATUSES, Event, Shipment, store


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local-only default matching .env.example.
    mock_shipping_api_key: str = "mock-shipping-local-key"


settings = Settings()

app = FastAPI(title="Mock shipping provider")
add_error_handler(app)
api = APIRouter(dependencies=[Depends(api_key_check(settings.mock_shipping_api_key))])

Status = Literal["label_created", "in_transit", "out_for_delivery", "delivered", "exception"]
assert set(Status.__args__) == set(STATUSES)


class EventModel(BaseModel):
    status: Status
    at: datetime
    location: str


class ShipmentOut(BaseModel):
    tracking_id: str
    order_id: str
    status: str
    estimated_delivery: date
    delivered_at: datetime | None
    # When the courier last updated the shipment (shipping_delays_v1 section 5).
    last_updated_at: datetime
    events: list[EventModel]


def shipment_out(shipment: Shipment) -> ShipmentOut:
    return ShipmentOut(
        tracking_id=shipment.tracking_id,
        order_id=shipment.order_id,
        status=shipment.status,
        estimated_delivery=shipment.estimated_delivery,
        delivered_at=shipment.delivered_at,
        last_updated_at=shipment.events[-1].at,
        events=[EventModel(**e.__dict__) for e in shipment.events],
    )


def apply_read_fault(order_id: str) -> None:
    fault = store.faults.take("shipment.read", queued=True, order_id=order_id)
    if fault is None:
        return
    if fault.mode == "error":
        raise_fault_error(fault)
    # mode=delay: hold the response, outside the lock so other requests are not blocked.
    time.sleep(fault.delay_seconds)


# --- Admin ------------------------------------------------------------------------------


class SeedShipment(BaseModel):
    tracking_id: str
    order_id: str
    status: Status
    estimated_delivery: date
    delivered_at: datetime | None = None
    events: list[EventModel] = Field(min_length=1)


class ResetRequest(BaseModel):
    shipments: list[SeedShipment]


@api.post("/admin/reset")
def reset(request: ResetRequest) -> dict[str, int]:
    """Replace all state with the seed data and clear queued faults."""
    shipments = [
        Shipment(
            tracking_id=s.tracking_id,
            order_id=s.order_id,
            status=s.status,
            estimated_delivery=s.estimated_delivery,
            delivered_at=s.delivered_at,
            events=[Event(**e.model_dump()) for e in s.events],
        )
        for s in request.shipments
    ]
    store.reset(shipments)
    return {"shipments": len(shipments)}


class NewEvent(BaseModel):
    status: Status
    location: str = Field(min_length=1)
    # Defaults to now; pass a time to keep evals deterministic.
    at: datetime | None = None
    # A new delivery estimate from the courier, if it gave one (shipping_delays_v1 section 5).
    estimated_delivery: date | None = None


@api.post("/admin/shipments/{tracking_id}/events")
def add_event(tracking_id: str, request: NewEvent) -> ShipmentOut:
    """Record a courier scan, e.g. an `exception` or `out_for_delivery`."""
    with store.lock:
        shipment = store.shipments.get(tracking_id)
        if shipment is None:
            raise MockError(status.HTTP_404_NOT_FOUND, "SHIPMENT_NOT_FOUND", "No such shipment.")
        if shipment.status == "delivered":
            raise MockError(
                status.HTTP_409_CONFLICT, "SHIPMENT_DELIVERED", "The shipment is already delivered."
            )
        at = request.at or datetime.now(UTC)
        if at < shipment.events[-1].at:
            raise MockError(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "EVENT_OUT_OF_ORDER",
                "A new event cannot be older than the last one.",
            )
        shipment.events.append(Event(status=request.status, at=at, location=request.location))
        shipment.status = request.status
        if request.status == "delivered":
            shipment.delivered_at = at
        if request.estimated_delivery is not None:
            shipment.estimated_delivery = request.estimated_delivery
        return shipment_out(shipment)


# Both shipment reads are one logical operation, matched by order_id.
OPERATIONS = frozenset({"shipment.read"})


class FaultRequest(FaultRequestBase):
    """Applied to the next `count` shipment reads, in order.

    error: the read fails with this error. delay: the response is held `delay_seconds`.
    """

    mode: Literal["error", "delay"]


add_fault_routes(api, store.faults, FaultRequest, OPERATIONS)


@app.get("/health")
def health() -> dict[str, str]:
    """Public, for the docker healthcheck."""
    return {"status": "ok"}


# --- Queries ----------------------------------------------------------------------------


@api.get("/shipments/{tracking_id}")
def get_shipment(tracking_id: str) -> ShipmentOut:
    with store.lock:
        known = store.shipments.get(tracking_id)
    apply_read_fault(known.order_id if known else "")
    with store.lock:
        shipment = store.shipments.get(tracking_id)
        if shipment is None:
            raise MockError(status.HTTP_404_NOT_FOUND, "SHIPMENT_NOT_FOUND", "No such shipment.")
        return shipment_out(shipment)


@api.get("/shipments")
def find_shipments(order_id: Annotated[str, Query(min_length=1)]) -> list[ShipmentOut]:
    """Shipments for an order; an empty list when the courier has none."""
    apply_read_fault(order_id)
    with store.lock:
        return [shipment_out(s) for s in store.shipments.values() if s.order_id == order_id]


app.include_router(api)
