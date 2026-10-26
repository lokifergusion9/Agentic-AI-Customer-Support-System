"""In-memory state of the mock ticketing (helpdesk) provider.

State is lost on restart; `POST /admin/reset` loads it again from the seed data
(`scripts.seed_data.mock_ticketing_state`). One lock guards the tickets; faults have their own.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime

from mock_services.common import FaultQueue

# The same statuses as the app's tickets table.
STATUSES = ("open", "awaiting_customer", "awaiting_approval", "escalated", "resolved")


@dataclass
class Note:
    text: str
    at: datetime


@dataclass
class Update:
    """One applied update, kept so a retry with the same update_id is not applied twice."""

    update_id: str
    ticket_ref: str
    status: str | None
    note: str | None
    applied_at: datetime


@dataclass
class Ticket:
    ref: str
    local_ticket_id: str
    customer_id: str
    subject: str
    status: str
    created_at: datetime
    updated_at: datetime
    notes: list[Note] = field(default_factory=list)


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tickets: dict[str, Ticket] = {}  # by ref
        self.by_local_id: dict[str, Ticket] = {}  # by local_ticket_id
        self.updates: dict[str, Update] = {}  # by update_id
        self.faults = FaultQueue()  # applied to new creates and updates

    def reset(self, tickets: list[Ticket]) -> None:
        with self.lock:
            self.tickets = {t.ref: t for t in tickets}
            self.by_local_id = {t.local_ticket_id: t for t in tickets}
            self.updates = {}
            self.faults.clear()

    def add(self, ticket: Ticket) -> None:
        """Record a new ticket. The caller holds `lock`."""
        self.tickets[ticket.ref] = ticket
        self.by_local_id[ticket.local_ticket_id] = ticket


store = Store()
