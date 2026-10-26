"""In-memory state of the mock shipping provider.

State is lost on restart; `POST /admin/reset` loads it again from the seed data
(`scripts.seed_data.mock_shipping_state`). One lock guards the shipments; faults have their own.
"""

import threading
from dataclasses import dataclass, field
from datetime import date, datetime

from mock_services.common import FaultQueue

# The courier statuses in shipping_delays_v1 section 3.
STATUSES = ("label_created", "in_transit", "out_for_delivery", "delivered", "exception")


@dataclass
class Event:
    status: str
    at: datetime
    location: str


@dataclass
class Shipment:
    tracking_id: str
    order_id: str
    status: str
    estimated_delivery: date
    delivered_at: datetime | None
    # Oldest first; the last event is the current status.
    events: list[Event] = field(default_factory=list)


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.shipments: dict[str, Shipment] = {}  # by tracking_id
        self.faults = FaultQueue()  # applied to shipment reads

    def reset(self, shipments: list[Shipment]) -> None:
        with self.lock:
            self.shipments = {s.tracking_id: s for s in shipments}
            self.faults.clear()


store = Store()
