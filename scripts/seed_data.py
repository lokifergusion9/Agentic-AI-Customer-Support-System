"""Seed synthetic customers, orders, order items, returns and past refunds (W1.2).

    uv run python -m scripts.seed_data                     # dates relative to today
    uv run python -m scripts.seed_data --today 2026-10-25  # fixed date (tests, evals)
    uv run python -m scripts.seed_data --push-mocks        # also reset the mock services

Business tables are emptied first; policy tables are left alone (they belong to W2.1).
Scenario dates are offsets from `today`, so each scenario behaves the same on whichever day it
is seeded. The one exception is the 2025 order, which needs real 2025 dates to fall under
refund_policy v0. Shipping delays count business days, as the shipping policy does.

The data for the mock payment and shipping services is defined here too, so there is a single
source of truth. The mocks receive it through their reset endpoints (W1.8-W1.10) when
`--push-mocks` is given; their state is in memory, so push again after restarting them.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import Connection, create_engine, text

from app.config import get_settings
from app.db import SCHEMA

IST = timezone(timedelta(hours=5, minutes=30))
CURRENCY = "INR"
SEED_REVIEWER = "auth|reviewer_riya"

# Emptied on every seed. Policy tables are not listed and are kept.
BUSINESS_TABLES = [
    "audit_events",
    "refund_operations",
    "approvals",
    "action_proposals",
    "workflow_runs",
    "tickets",
    "returns",
    "order_items",
    "orders",
    "customers",
]

CUSTOMERS = [
    ("cus_alice", "auth|alice", "Alice Mathew", "alice@example.test"),
    ("cus_bob", "auth|bob", "Bob Fernandes", "bob@example.test"),
    ("cus_chitra", "auth|chitra", "Chitra Iyer", "chitra@example.test"),
    ("cus_dev", "auth|dev", "Dev Malhotra", "dev@example.test"),
]


@dataclass(frozen=True)
class Item:
    product_id: str
    name: str
    amount_minor: int
    quantity: int = 1


@dataclass(frozen=True)
class Return:
    status: str
    requested: date
    received: date | None = None
    inspection: str | None = None


@dataclass(frozen=True)
class Shipment:
    status: str
    estimated_delivery: date


@dataclass(frozen=True)
class Scenario:
    name: str
    order_id: str
    customer_id: str
    status: str
    placed: date
    items: list[Item]
    delivered: date | None = None
    shipping_minor: int = 0
    return_: Return | None = None
    # Status of a refund already issued for this order: succeeded or submitted (pending).
    past_refund: str | None = None
    # Defaults to a shipment matching the order dates.
    shipment: Shipment | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def paid_amount_minor(self) -> int:
        return sum(item.amount_minor * item.quantity for item in self.items) + self.shipping_minor

    @property
    def number(self) -> str:
        return self.order_id.removeprefix("ord_")


def business_days_before(day: date, count: int) -> date:
    while count:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            count -= 1
    return day


def at(day: date, hour: int = 10) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=IST)


def scenarios(today: date) -> list[Scenario]:
    def ago(days: int) -> date:
        return today - timedelta(days=days)

    def delayed(order_id: str, customer: str, item: Item, days_late: int, name: str) -> Scenario:
        estimate = business_days_before(today, days_late)
        return Scenario(
            name,
            order_id,
            customer,
            "shipped",
            placed=estimate - timedelta(days=7),
            items=[item],
            shipment=Shipment("in_transit", estimate),
        )

    return [
        Scenario(
            "eligible_refund",
            "ord_1001",
            "cus_alice",
            "delivered",
            placed=ago(13),
            delivered=ago(9),
            items=[Item("prd_headphones", "Wireless Headphones", 249900)],
            return_=Return("received", ago(7), received=ago(3), inspection="passed"),
            notes=["The PDF reference case: returned last week, no refund yet."],
        ),
        Scenario(
            "already_refunded",
            "ord_1002",
            "cus_alice",
            "delivered",
            placed=ago(30),
            delivered=ago(26),
            items=[Item("prd_speaker", "Bluetooth Speaker", 399900)],
            return_=Return("closed", ago(24), received=ago(20), inspection="passed"),
            past_refund="succeeded",
        ),
        Scenario(
            "refund_processing",
            "ord_1003",
            "cus_bob",
            "delivered",
            placed=ago(16),
            delivered=ago(12),
            items=[Item("prd_smartwatch", "Smartwatch", 1299900)],
            return_=Return("received", ago(10), received=ago(5), inspection="passed"),
            past_refund="submitted",
            notes=["Also the target of cross-customer access tests (owned by Bob)."],
        ),
        Scenario(
            "return_not_received",
            "ord_1004",
            "cus_bob",
            "delivered",
            placed=ago(10),
            delivered=ago(6),
            items=[Item("prd_charger", "USB-C Charger", 129900)],
            return_=Return("in_transit", ago(3)),
        ),
        Scenario(
            "inspection_failed",
            "ord_1005",
            "cus_chitra",
            "delivered",
            placed=ago(14),
            delivered=ago(10),
            items=[Item("prd_mouse", "Gaming Mouse", 219900)],
            return_=Return("received", ago(8), received=ago(4), inspection="failed"),
        ),
        Scenario(
            "inspection_pending",
            "ord_1006",
            "cus_chitra",
            "delivered",
            placed=ago(9),
            delivered=ago(6),
            items=[Item("prd_keyboard", "Mechanical Keyboard", 449900)],
            return_=Return("received", ago(4), received=ago(1), inspection="pending"),
        ),
        Scenario(
            "damaged_item",
            "ord_1007",
            "cus_dev",
            "delivered",
            placed=ago(4),
            delivered=ago(1),
            items=[Item("prd_monitor", "27-inch Monitor", 1499900)],
            shipping_minor=9900,
            notes=["Reported within the 48-hour window; shipping is refundable."],
        ),
        delayed(
            "ord_1008",
            "cus_alice",
            Item("prd_powerbank", "Power Bank", 179900),
            2,
            "minor_delay",
        ),
        delayed(
            "ord_1009",
            "cus_bob",
            Item("prd_earbuds", "Wireless Earbuds", 349900),
            5,
            "significant_delay",
        ),
        delayed(
            "ord_1010",
            "cus_chitra",
            Item("prd_tablet", "10-inch Tablet", 2499900),
            8,
            "possibly_lost",
        ),
        Scenario(
            "order_status_delivered",
            "ord_1011",
            "cus_dev",
            "delivered",
            placed=ago(8),
            delivered=ago(4),
            items=[Item("prd_phonecase", "Phone Case", 59900)],
        ),
        Scenario(
            "outside_return_window",
            "ord_1012",
            "cus_dev",
            "delivered",
            placed=ago(19),
            delivered=ago(15),
            items=[Item("prd_webcam", "HD Webcam", 329900)],
        ),
        Scenario(
            "policy_v0_window",
            "ord_1013",
            "cus_alice",
            "delivered",
            # Absolute dates: refund_policy v0 (7-day window) was in effect in 2025.
            placed=date(2025, 6, 6),
            delivered=date(2025, 6, 10),
            items=[Item("prd_smartspeaker", "Smart Speaker", 499900)],
            return_=Return(
                "received", date(2025, 6, 18), received=date(2025, 6, 22), inspection="passed"
            ),
            notes=["Requested 8 days after delivery: ineligible under v0, eligible under v1."],
        ),
        Scenario(
            "high_value_refund",
            "ord_1014",
            "cus_bob",
            "delivered",
            placed=ago(12),
            delivered=ago(8),
            items=[Item("prd_laptop", "14-inch Laptop", 8999900)],
            return_=Return("received", ago(6), received=ago(2), inspection="passed"),
            notes=["Above ₹50,000: needs a senior_reviewer."],
        ),
        Scenario(
            "partial_return",
            "ord_1015",
            "cus_chitra",
            "delivered",
            placed=ago(9),
            delivered=ago(5),
            items=[
                Item("prd_camera", "Mirrorless Camera", 4599900),
                Item("prd_lens", "50mm Lens", 1299900),
                Item("prd_camerabag", "Camera Bag", 199900),
            ],
            notes=["Customer wants to return only the lens: out of scope."],
        ),
    ]


def payload_hash(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of a proposal payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def shipment(scenario: Scenario) -> Shipment:
    if scenario.shipment:
        return scenario.shipment
    assert scenario.delivered, f"{scenario.order_id} needs an explicit shipment"
    return Shipment("delivered", scenario.delivered)


def mock_payments_state(today: date) -> list[dict[str, Any]]:
    """Payments and refunds held by the mock payment provider (W1.8)."""
    payments = []
    for scenario in scenarios(today):
        refunds = []
        if scenario.past_refund:
            refunds.append(
                {
                    "provider_reference": f"rf_{scenario.number}",
                    "idempotency_key": f"refund:prp_{scenario.number}",
                    "amount_minor": scenario.paid_amount_minor,
                    "status": "succeeded" if scenario.past_refund == "succeeded" else "pending",
                }
            )
        payments.append(
            {
                "payment_id": f"pay_{scenario.number}",
                "order_id": scenario.order_id,
                "amount_minor": scenario.paid_amount_minor,
                "currency": CURRENCY,
                "status": "captured",
                "captured_at": at(scenario.placed).isoformat(),
                "refunds": refunds,
            }
        )
    return payments


def courier_events(scenario: Scenario) -> list[dict[str, str]]:
    """Courier scans, oldest first. The last one is the current status (shipping_delays §3, §5).

    A delayed parcel's last scan is the day before its estimate, so a possibly lost parcel
    has had no update for over a week.
    """
    current = shipment(scenario)

    def event(status: str, day: date, hour: int, location: str) -> dict[str, str]:
        return {"status": status, "at": at(day, hour).isoformat(), "location": location}

    events = [
        event("label_created", scenario.placed, 18, "Bengaluru warehouse"),
        event("in_transit", scenario.placed + timedelta(days=1), 9, "Bengaluru sorting hub"),
    ]
    if current.status == "delivered":
        assert scenario.delivered
        events += [
            event("out_for_delivery", scenario.delivered, 7, "Local delivery centre"),
            # Same time as orders.delivered_at.
            event("delivered", scenario.delivered, 10, "Delivered to customer"),
        ]
    else:
        last_scan = current.estimated_delivery - timedelta(days=1)
        events.append(event(current.status, last_scan, 21, "Destination hub"))
    return events


def mock_shipping_state(today: date) -> list[dict[str, Any]]:
    """Shipments held by the mock shipping provider (W1.9)."""
    shipments = []
    for scenario in scenarios(today):
        current = shipment(scenario)
        shipments.append(
            {
                "tracking_id": f"trk_{scenario.number}",
                "order_id": scenario.order_id,
                "status": current.status,
                "estimated_delivery": current.estimated_delivery.isoformat(),
                "delivered_at": at(scenario.delivered).isoformat() if scenario.delivered else None,
                "events": courier_events(scenario),
            }
        )
    return shipments


def mock_ticketing_state(today: date) -> list[dict[str, Any]]:
    """Tickets held by the mock ticketing provider (W1.10): the earlier refund tickets.

    They match the tickets inserted by `insert_past_refund`, linked by external_ticket_ref.
    """
    tickets = []
    for scenario in scenarios(today):
        if not scenario.past_refund:
            continue
        assert scenario.return_ and scenario.return_.received
        handled_on = scenario.return_.received
        tickets.append(
            {
                "ref": f"ext_{scenario.number}",
                "local_ticket_id": f"tkt_{scenario.number}",
                "customer_id": scenario.customer_id,
                "subject": f"Refund request for {scenario.order_id}",
                "status": past_ticket_status(scenario),
                "created_at": at(handled_on, 9).isoformat(),
                "notes": [
                    {"text": past_ticket_note(scenario), "at": at(handled_on, 11).isoformat()}
                ],
            }
        )
    return tickets


def seed(connection: Connection, today: date) -> list[Scenario]:
    """Replace all business data with the scenarios as of `today`."""
    tables = ", ".join(f"{SCHEMA}.{table}" for table in BUSINESS_TABLES)
    connection.execute(text(f"TRUNCATE {tables}"))

    connection.execute(
        text(
            f"INSERT INTO {SCHEMA}.customers (id, external_auth_subject, name, email) "
            "VALUES (:id, :subject, :name, :email)"
        ),
        [dict(zip(("id", "subject", "name", "email"), row, strict=True)) for row in CUSTOMERS],
    )

    all_scenarios = scenarios(today)
    for scenario in all_scenarios:
        insert_order(connection, scenario)
        if scenario.past_refund:
            insert_past_refund(connection, scenario)
    return all_scenarios


def insert_order(connection: Connection, scenario: Scenario) -> None:
    connection.execute(
        text(
            f"INSERT INTO {SCHEMA}.orders (id, customer_id, status, currency, paid_amount_minor, "
            "shipping_amount_minor, placed_at, delivered_at) VALUES (:id, :customer_id, :status, "
            ":currency, :paid, :shipping, :placed_at, :delivered_at)"
        ),
        {
            "id": scenario.order_id,
            "customer_id": scenario.customer_id,
            "status": scenario.status,
            "currency": CURRENCY,
            "paid": scenario.paid_amount_minor,
            "shipping": scenario.shipping_minor,
            "placed_at": at(scenario.placed),
            "delivered_at": at(scenario.delivered) if scenario.delivered else None,
        },
    )
    connection.execute(
        text(
            f"INSERT INTO {SCHEMA}.order_items (id, order_id, product_id, product_name, quantity, "
            "paid_amount_minor) VALUES (:id, :order_id, :product_id, :name, :quantity, :amount)"
        ),
        [
            {
                "id": f"itm_{scenario.number}_{index}",
                "order_id": scenario.order_id,
                "product_id": item.product_id,
                "name": item.name,
                "quantity": item.quantity,
                "amount": item.amount_minor * item.quantity,
            }
            for index, item in enumerate(scenario.items, start=1)
        ],
    )
    if scenario.return_:
        connection.execute(
            text(
                f"INSERT INTO {SCHEMA}.returns (id, order_id, status, requested_at, received_at, "
                "inspection_status) VALUES (:id, :order_id, :status, :requested_at, "
                ":received_at, :inspection)"
            ),
            {
                "id": f"ret_{scenario.number}",
                "order_id": scenario.order_id,
                "status": scenario.return_.status,
                "requested_at": at(scenario.return_.requested),
                "received_at": (
                    at(scenario.return_.received) if scenario.return_.received else None
                ),
                "inspection": scenario.return_.inspection,
            },
        )


def past_ticket_status(scenario: Scenario) -> str:
    return "resolved" if scenario.past_refund == "succeeded" else "open"


def past_ticket_note(scenario: Scenario) -> str:
    if scenario.past_refund == "succeeded":
        return "Refund issued."
    return "Refund submitted, awaiting provider."


def insert_past_refund(connection: Connection, scenario: Scenario) -> None:
    """A refund handled by an earlier ticket: ticket, run, proposal, approval, operation."""
    assert scenario.return_ and scenario.return_.received
    number = scenario.number
    approved_on = scenario.return_.received
    payload = {
        "action": "request_refund_approval",
        "order_id": scenario.order_id,
        "amount_minor": scenario.paid_amount_minor,
        "currency": CURRENCY,
        "reason": "Return received and inspection passed",
        "evidence_refs": [f"return_ret_{number}", f"payment_pay_{number}"],
        "policy_refs": ["refund_policy_v1:section_3"],
        "missing_information": [],
    }
    digest = payload_hash(payload)
    statements = [
        (
            f"INSERT INTO {SCHEMA}.tickets (id, customer_id, order_id, category, status, "
            "customer_message, summary, external_ticket_ref, created_at, updated_at) VALUES "
            "(:ticket, :customer, :order, 'refund_request', :ticket_status, "
            "'I returned my item. When will I get my refund?', :summary, :external_ref, "
            ":started, :completed)"
        ),
        (
            f"INSERT INTO {SCHEMA}.workflow_runs (id, ticket_id, thread_id, status, started_at, "
            "completed_at) VALUES (:run, :ticket, :thread, 'completed', :started, :completed)"
        ),
        (
            f"INSERT INTO {SCHEMA}.action_proposals (id, run_id, order_id, customer_id, "
            "action_type, payload, payload_hash, status) VALUES (:proposal, :run, :order, "
            ":customer, 'refund', CAST(:payload AS jsonb), :hash, 'executed')"
        ),
        (
            f"INSERT INTO {SCHEMA}.approvals (id, proposal_id, approved_payload_hash, "
            "required_role, reviewer_id, reviewer_role, decision, expires_at, decided_at) VALUES "
            "(:approval, :proposal, :hash, 'reviewer', :reviewer, 'reviewer', 'approved', "
            ":expires, :decided)"
        ),
        (
            f"INSERT INTO {SCHEMA}.refund_operations (id, order_id, proposal_id, approval_id, "
            "amount_minor, idempotency_key, provider_reference, status) VALUES (:operation, "
            ":order, :proposal, :approval, :amount, :key, :reference, :operation_status)"
        ),
    ]
    params = {
        "ticket": f"tkt_{number}",
        "run": f"run_{number}",
        "thread": f"th_{number}",
        "proposal": f"prp_{number}",
        "approval": f"apr_{number}",
        "operation": f"rop_{number}",
        "customer": scenario.customer_id,
        "order": scenario.order_id,
        "ticket_status": past_ticket_status(scenario),
        "summary": past_ticket_note(scenario),
        "external_ref": f"ext_{number}",
        "started": at(approved_on, 9),
        "completed": at(approved_on, 11),
        "payload": json.dumps(payload),
        "hash": digest,
        "reviewer": SEED_REVIEWER,
        "decided": at(approved_on, 10),
        "expires": at(approved_on, 11),
        "amount": scenario.paid_amount_minor,
        "key": f"refund:prp_{number}",
        "reference": f"rf_{number}",
        "operation_status": scenario.past_refund,
    }
    for statement in statements:
        connection.execute(text(statement), params)


def push_mock_payments(client: httpx.Client, today: date) -> dict[str, int]:
    """Replace the mock payment provider's state with the payments as of `today`."""
    response = client.post("/admin/reset", json={"payments": mock_payments_state(today)})
    response.raise_for_status()
    return response.json()


def push_mock_shipping(client: httpx.Client, today: date) -> dict[str, int]:
    """Replace the mock shipping provider's state with the shipments as of `today`."""
    response = client.post("/admin/reset", json={"shipments": mock_shipping_state(today)})
    response.raise_for_status()
    return response.json()


def push_mock_ticketing(client: httpx.Client, today: date) -> dict[str, int]:
    """Replace the mock ticketing provider's state with the tickets as of `today`."""
    response = client.post("/admin/reset", json={"tickets": mock_ticketing_state(today)})
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=datetime.now(IST).date(),
        help="date the scenarios are relative to (default: today)",
    )
    parser.add_argument(
        "--push-mocks",
        action="store_true",
        help="also load the mock payment, shipping and ticketing data (they must be running)",
    )
    args = parser.parse_args()
    settings = get_settings()

    engine = create_engine(settings.database_url)
    with engine.begin() as connection:
        seeded = seed(connection, args.today)
    engine.dispose()

    print(f"Seeded {len(CUSTOMERS)} customers and {len(seeded)} orders as of {args.today}:")
    for scenario in seeded:
        print(f"  {scenario.order_id}  {scenario.customer_id:<11} {scenario.name}")

    if args.push_mocks:
        mocks = [
            (
                "payment",
                settings.mock_payments_url,
                settings.mock_payments_api_key,
                push_mock_payments,
            ),
            (
                "shipping",
                settings.mock_shipping_url,
                settings.mock_shipping_api_key,
                push_mock_shipping,
            ),
            (
                "ticketing",
                settings.mock_ticketing_url,
                settings.mock_ticketing_api_key,
                push_mock_ticketing,
            ),
        ]
        for name, url, api_key, push in mocks:
            client = httpx.Client(base_url=url, headers={"X-API-Key": api_key}, timeout=10)
            try:
                with client:
                    pushed = push(client, args.today)
            except httpx.HTTPError as error:
                raise SystemExit(
                    f"Could not reset the mock {name} API at {url}: {error}"
                ) from error
            print(f"Mock {name}: " + ", ".join(f"{n} {kind}" for kind, n in pushed.items()))


if __name__ == "__main__":
    main()
