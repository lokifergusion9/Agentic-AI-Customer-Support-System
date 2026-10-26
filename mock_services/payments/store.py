"""In-memory state of the mock payment provider.

State is lost on restart; `POST /admin/reset` loads it again from the seed data
(`scripts.seed_data.mock_payments_state`). One lock guards every read and write, so
concurrent requests see a consistent view.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Refund:
    provider_reference: str
    idempotency_key: str
    order_id: str
    amount_minor: int
    currency: str
    status: str  # pending | succeeded | failed
    created_at: datetime


@dataclass
class Payment:
    payment_id: str
    order_id: str
    amount_minor: int
    currency: str
    status: str
    captured_at: datetime
    refunds: list[Refund] = field(default_factory=list)

    @property
    def refundable_minor(self) -> int:
        """Paid amount minus succeeded and pending refunds."""
        held = sum(r.amount_minor for r in self.refunds if r.status in ("pending", "succeeded"))
        return self.amount_minor - held


@dataclass(frozen=True)
class Fault:
    """An injected failure for one new refund request (see POST /admin/faults)."""

    mode: str  # error | delay | accept_then_timeout
    status_code: int
    error_code: str
    retryable: bool
    delay_seconds: float


class Store:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.payments: dict[str, Payment] = {}  # by order_id
        self.refunds: dict[str, Refund] = {}  # by provider_reference
        self.by_key: dict[str, Refund] = {}  # by idempotency_key
        self.faults: list[Fault] = []  # applied to new refund requests, oldest first

    def reset(self, payments: list[Payment]) -> None:
        with self.lock:
            self.payments = {p.order_id: p for p in payments}
            self.refunds = {r.provider_reference: r for p in payments for r in p.refunds}
            self.by_key = {r.idempotency_key: r for r in self.refunds.values()}
            self.faults = []

    def add(self, payment: Payment, refund: Refund) -> None:
        """Record a new refund. The caller holds `lock`."""
        payment.refunds.append(refund)
        self.refunds[refund.provider_reference] = refund
        self.by_key[refund.idempotency_key] = refund


store = Store()
