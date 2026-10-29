"""A clean environment for every attempt (W4.4, E§7, Q11).

    with environment_lock(connection):
        env = prepare(connection, mocks, task)        # reset, load the fixture, arm faults
        ...                                           # the adapter runs the agent (W4.7)
        capture(connection, mocks, env, attempt_id, workflow_run_id)   # snapshot + trace

Attempts run one at a time: a session-level PostgreSQL advisory lock is held for the whole
attempt, so two workers never share the business tables. Each attempt empties the business
tables (never `jobs`, the policy tables or the `eval` schema), resets the three mocks, and
loads only its own fixture, through the same inserts the W1.2 seed uses.

The clock is not global: `env.as_of` is handed to the workflow as its `today` (W4.7).

Faults are keyed to a logical operation and, for payments and shipping, an order (E§22), so
two configurations that call the providers in a different order still meet the same failure.
"""

import hashlib
import json
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx
from sqlalchemy import Connection, text

from app.config import get_settings
from app.db import SCHEMA
from evals.db import EVAL_SCHEMA
from evals.task_format import EnvironmentFixture, EvaluatorView, FaultFixture
from scripts.seed_data import (
    BUSINESS_TABLES,
    Item,
    Return,
    Scenario,
    Shipment,
    insert_customers,
    insert_order,
    insert_past_refund,
    payments_state,
    shipping_state,
    ticketing_state,
)

# Emptied before every attempt. `jobs` stays: the eval_attempt job running this attempt is in it.
RESET_TABLES = [table for table in BUSINESS_TABLES if table != "jobs"]
# Copied into the final-state snapshot, in this order.
SNAPSHOT_TABLES = [
    "customers",
    "orders",
    "order_items",
    "returns",
    "tickets",
    "workflow_runs",
    "action_proposals",
    "approvals",
    "refund_operations",
]
LOCK_NAME = "eval:environment"

# Fixture operation -> (mock, logical operation in that mock).
FAULT_TARGETS = {
    "payments.get_payment": ("payments", "payment.get"),
    "payments.create_refund": ("payments", "refund.create"),
    "shipping.get_shipment": ("shipping", "shipment.read"),
    "ticketing.update_ticket": ("ticketing", "ticket.update"),
}


@dataclass(frozen=True)
class MockClients:
    """httpx clients for the three mocks, each sending its X-API-Key header."""

    payments: httpx.Client
    shipping: httpx.Client
    ticketing: httpx.Client

    def get(self, name: str) -> httpx.Client:
        return getattr(self, name)


@dataclass(frozen=True)
class Environment:
    environment_id: str
    task_id: str
    as_of: date


@contextmanager
def environment_lock(connection: Connection) -> Iterator[None]:
    """Hold the environment for one attempt. Blocks while another attempt holds it."""
    connection.execute(
        text("SELECT pg_advisory_lock(hashtextextended(:name, 0))"), {"name": LOCK_NAME}
    )
    try:
        yield
    finally:
        connection.execute(
            text("SELECT pg_advisory_unlock(hashtextextended(:name, 0))"), {"name": LOCK_NAME}
        )


def prepare(
    connection: Connection,
    mocks: MockClients,
    task: EvaluatorView,
    *,
    timeout_fault_seconds: float | None = None,
) -> Environment:
    """Reset everything the agent can see and load this task's fixture. Commits nothing itself.

    `timeout_fault_seconds` is how long a timeout fault holds a response: longer than the
    provider client's timeout by default, so the client really times out.
    """
    fixture = task.environment_fixture
    all_scenarios = scenarios_for(fixture)

    tables = ", ".join(f"{SCHEMA}.{table}" for table in RESET_TABLES)
    connection.execute(text(f"TRUNCATE {tables}"))
    insert_customers(
        connection,
        [(c.id, c.external_auth_subject, c.name, c.email) for c in fixture.customers],
    )
    for scenario in all_scenarios:
        insert_order(connection, scenario)
        if scenario.past_refund:
            insert_past_refund(connection, scenario)

    shipped = {s.order_id for s in fixture.shipments}
    reset_mock(mocks.payments, {"payments": payments_state(all_scenarios)})
    reset_mock(
        mocks.shipping,
        {"shipments": shipping_state([s for s in all_scenarios if s.order_id in shipped])},
    )
    reset_mock(mocks.ticketing, {"tickets": ticketing_state(all_scenarios)})

    hold = timeout_fault_seconds or get_settings().provider_timeout_seconds + 1
    for fault in fixture.faults:
        name, request = fault_request(fault, hold)
        response = mocks.get(name).post("/admin/faults", json=request)
        response.raise_for_status()

    return Environment(
        environment_id=f"env_{secrets.token_hex(8)}", task_id=task.task_id, as_of=fixture.as_of
    )


def scenarios_for(fixture: EnvironmentFixture) -> list[Scenario]:
    """The fixture as W1.2 seed scenarios, so it is loaded exactly the way the seed is."""
    returns = one_per_order("returns", fixture.returns)
    payments = one_per_order("payments", fixture.payments)
    shipments = one_per_order("shipments", fixture.shipments)
    result = []
    for order in fixture.orders:
        payment = payments.get(order.id)
        if payment is None or payment.amount_minor != order.paid_amount_minor:
            raise ValueError(f"{order.id}: needs one payment for the paid amount")
        if len(payment.refunds) > 1 or any(
            r.amount_minor != order.paid_amount_minor for r in payment.refunds
        ):
            raise ValueError(f"{order.id}: past refunds must be one full refund")
        ret, shipment = returns.get(order.id), shipments.get(order.id)
        result.append(
            Scenario(
                name=order.id,
                order_id=order.id,
                customer_id=order.customer_id,
                status=order.status,
                placed=order.placed_at,
                delivered=order.delivered_at,
                items=[Item(i.product_id, i.name, i.amount_minor, i.quantity) for i in order.items],
                shipping_minor=order.shipping_minor,
                return_=(
                    Return(ret.status, ret.requested_at, ret.received_at, ret.inspection_status)
                    if ret
                    else None
                ),
                past_refund=payment.refunds[0].status if payment.refunds else None,
                shipment=(
                    Shipment(shipment.status, shipment.estimated_delivery) if shipment else None
                ),
            )
        )
    return result


def one_per_order(kind: str, rows: list[Any]) -> dict[str, Any]:
    by_order = {row.order_id: row for row in rows}
    if len(by_order) != len(rows):
        raise ValueError(f"at most one of {kind} per order")
    return by_order


def fault_request(fault: FaultFixture, hold_seconds: float) -> tuple[str, dict[str, Any]]:
    """The mock and the POST /admin/faults body for a fixture fault."""
    name, operation = FAULT_TARGETS[fault.operation]
    mode = {"http_5xx": "error", "timeout": "delay", "unknown_outcome": "accept_then_timeout"}
    request: dict[str, Any] = {
        "operation": operation,
        "mode": mode[fault.kind],
        "occurrence": fault.on_call,
        "count": fault.times,
        "delay_seconds": hold_seconds,
    }
    if fault.order_id is not None:
        request["match"] = {"order_id": fault.order_id}
    return name, request


def reset_mock(client: httpx.Client, state: dict[str, Any]) -> None:
    response = client.post("/admin/reset", json=state)
    response.raise_for_status()


# --- After the attempt -------------------------------------------------------------------------


def snapshot(connection: Connection, mocks: MockClients, env: Environment) -> dict[str, Any]:
    """The final business state: support rows plus what each mock provider holds."""
    support = {
        table: [
            dict(row._mapping)
            for row in connection.execute(text(f"SELECT * FROM {SCHEMA}.{table} ORDER BY id"))
        ]
        for table in SNAPSHOT_TABLES
    }
    order_ids = [row["id"] for row in support["orders"]]
    ticket_refs = [t["external_ticket_ref"] for t in support["tickets"] if t["external_ticket_ref"]]
    providers = {
        "payments": [found(mocks.payments, f"/payments/{oid}") for oid in order_ids],
        "refunds": [
            refund
            for oid in order_ids
            for refund in mocks.payments.get("/refunds", params={"order_id": oid}).json()
        ],
        "shipments": [
            shipment
            for oid in order_ids
            for shipment in mocks.shipping.get("/shipments", params={"order_id": oid}).json()
        ],
        "tickets": [found(mocks.ticketing, f"/tickets/{ref}") for ref in ticket_refs],
    }
    state = {
        "environment_id": env.environment_id,
        "task_id": env.task_id,
        "as_of": env.as_of,
        "support": support,
        "providers": {k: [v for v in rows if v is not None] for k, rows in providers.items()},
    }
    # Dates, datetimes and Decimals as JSON, the way the evaluators will read them.
    return json.loads(json.dumps(state, default=str))


def found(client: httpx.Client, path: str) -> Any:
    """GET a provider record, or None when the provider has no such record."""
    response = client.get(path)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def capture(
    connection: Connection,
    mocks: MockClients,
    env: Environment,
    attempt_id: str,
    workflow_run_id: str | None,
) -> dict[str, Any]:
    """Store the snapshot and copy the run's audit events into `eval` before the next reset."""
    state = snapshot(connection, mocks, env)
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"))
    connection.execute(
        text(
            f"INSERT INTO {EVAL_SCHEMA}.artifacts (id, attempt_id, kind, content, content_hash) "
            "VALUES (:id, :attempt, 'final_state', CAST(:content AS jsonb), :hash)"
        ),
        {
            "id": f"art_{secrets.token_hex(8)}",
            "attempt": attempt_id,
            "content": canonical,
            "hash": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        },
    )
    if workflow_run_id is not None:
        # The run's events, after the events of its ticket from before the run started.
        connection.execute(
            text(
                f"""
                INSERT INTO {EVAL_SCHEMA}.trace_events
                    (attempt_id, sequence_number, source_event_id, actor, event_type, payload,
                     occurred_at)
                SELECT :attempt, row_number() OVER (ORDER BY e.id), e.id, e.actor, e.event_type,
                       e.redacted_payload, e.created_at
                FROM {SCHEMA}.audit_events e
                WHERE e.run_id = :run
                   OR (e.run_id IS NULL AND e.ticket_id =
                       (SELECT ticket_id FROM {SCHEMA}.workflow_runs WHERE id = :run))
                """
            ),
            {"attempt": attempt_id, "run": workflow_run_id},
        )
    return state
