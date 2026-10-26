"""Typed result of the shipment read service (W1.12)."""

from datetime import date, datetime

from pydantic import BaseModel


class ShipmentEvent(BaseModel):
    status: str
    at: datetime
    location: str


class ShipmentView(BaseModel):
    order_id: str
    tracking_id: str
    # label_created | in_transit | out_for_delivery | delivered | exception (shipping_delays §3)
    status: str
    estimated_delivery: date
    delivered_at: datetime | None
    # When the courier last updated the shipment (shipping_delays §5).
    last_updated_at: datetime
    events: list[ShipmentEvent]
