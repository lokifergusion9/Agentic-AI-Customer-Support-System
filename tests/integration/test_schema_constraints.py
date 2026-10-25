"""Database rules hold even if application code gets them wrong (W1.3, W1.4).

Each test inserts rows inside a transaction that is rolled back afterwards.
"""

import hashlib
import threading
import time
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import DBAPIError

from app.db import SCHEMA

HASH_1 = hashlib.sha256(b"prp_1").hexdigest()
HASH_2 = hashlib.sha256(b"prp_2").hexdigest()

BASE_ROWS = [
    (
        "INSERT INTO support.customers (id, external_auth_subject, name, email) VALUES "
        "('cus_a', 'auth|a', 'Alice', 'alice@example.test'), "
        "('cus_b', 'auth|b', 'Bob', 'bob@example.test')"
    ),
    (
        "INSERT INTO support.orders "
        "(id, customer_id, status, currency, paid_amount_minor, placed_at, delivered_at) VALUES "
        "('ord_a', 'cus_a', 'delivered', 'INR', 249900, now(), now()), "
        "('ord_b', 'cus_b', 'delivered', 'INR', 100, now(), now())"
    ),
    (
        "INSERT INTO support.tickets (id, customer_id, order_id, status, customer_message) "
        "VALUES ('tkt_1', 'cus_a', 'ord_a', 'open', 'Where is my refund?')"
    ),
    (
        "INSERT INTO support.workflow_runs (id, ticket_id, thread_id, status) "
        "VALUES ('run_1', 'tkt_1', 'th_1', 'running')"
    ),
    (
        "INSERT INTO support.action_proposals "
        "(id, run_id, order_id, customer_id, action_type, payload, payload_hash, status) VALUES "
        f"('prp_1', 'run_1', 'ord_a', 'cus_a', 'refund', '{{}}', '{HASH_1}', 'approved'), "
        f"('prp_2', 'run_1', 'ord_a', 'cus_a', 'refund', '{{}}', '{HASH_2}', 'approved')"
    ),
    (
        "INSERT INTO support.approvals (id, proposal_id, approved_payload_hash, required_role, "
        "reviewer_id, reviewer_role, decision, expires_at, decided_at) VALUES "
        f"('apr_1', 'prp_1', '{HASH_1}', 'reviewer', 'rev_1', 'reviewer', 'approved', "
        "now() + interval '1 hour', now()), "
        f"('apr_2', 'prp_2', '{HASH_2}', 'reviewer', 'rev_1', 'reviewer', 'approved', "
        "now() + interval '1 hour', now())"
    ),
]

INSERT_REFUND = (
    "INSERT INTO support.refund_operations "
    "(id, order_id, proposal_id, approval_id, amount_minor, idempotency_key, status) "
    "VALUES (:id, :order_id, :proposal_id, :approval_id, :amount, :key, :status)"
)

INSERT_APPROVAL = (
    "INSERT INTO support.approvals (id, proposal_id, approved_payload_hash, required_role, "
    "reviewer_id, reviewer_role, decision, expires_at, decided_at) VALUES "
    "(:id, 'prp_1', :hash, :required_role, :reviewer_id, :reviewer_role, :decision, "
    "now() + make_interval(mins => :expires_in_minutes), "
    "CASE WHEN CAST(:decision AS text) IS NULL THEN NULL ELSE now() END)"
)


def insert_base_rows(connection: Connection) -> None:
    for statement in BASE_ROWS:
        connection.execute(text(statement))


def refund(**overrides: Any) -> dict[str, Any]:
    values = {
        "id": "rop_1",
        "order_id": "ord_a",
        "proposal_id": "prp_1",
        "approval_id": "apr_1",
        "amount": 100,
        "key": "refund:prp_1",
        "status": "reserved",
    }
    return values | overrides


def approval(**overrides: Any) -> dict[str, Any]:
    values = {
        "id": "apr_x",
        "hash": HASH_1,
        "required_role": "reviewer",
        "reviewer_id": None,
        "reviewer_role": None,
        "decision": None,
        "expires_in_minutes": 60,
    }
    return values | overrides


def rejected(conn: Connection, sql: str, params: dict[str, Any] | None = None) -> str:
    """Run a statement that must fail and return the violated constraint name or message."""
    with pytest.raises(DBAPIError) as info, conn.begin_nested():
        conn.execute(text(sql), params or {})
    error = info.value.orig
    return getattr(error.diag, "constraint_name", None) or str(error)


@pytest.fixture
def rows(conn: Connection) -> Connection:
    insert_base_rows(conn)
    return conn


# --- customers, orders, returns -----------------------------------------------------------


def test_one_customer_per_auth_subject(rows: Connection) -> None:
    sql = (
        "INSERT INTO support.customers (id, external_auth_subject, name, email) "
        "VALUES ('cus_c', 'auth|a', 'C', 'c@example.test')"
    )
    assert rejected(rows, sql) == "uq_customers_external_auth_subject"


@pytest.mark.parametrize(
    ("status", "currency", "amount", "delivered_at", "constraint"),
    [
        ("delivered", "INR", 100, None, "ck_orders_delivered_at"),
        ("placed", "INR", 100, "now()", "ck_orders_delivered_at"),
        ("placed", "inr", 100, None, "ck_orders_currency"),
        ("placed", "INR", -1, None, "ck_orders_paid_amount_minor"),
        ("lost", "INR", 100, None, "ck_orders_status"),
    ],
)
def test_invalid_orders_are_rejected(
    rows: Connection,
    status: str,
    currency: str,
    amount: int,
    delivered_at: str | None,
    constraint: str,
) -> None:
    sql = (
        "INSERT INTO support.orders "
        "(id, customer_id, status, currency, paid_amount_minor, placed_at, delivered_at) "
        f"VALUES ('ord_x', 'cus_a', :status, :currency, :amount, now(), {delivered_at or 'NULL'})"
    )
    params = {"status": status, "currency": currency, "amount": amount}
    assert rejected(rows, sql, params) == constraint


@pytest.mark.parametrize(
    ("status", "received_at", "inspection_status", "constraint"),
    [
        ("received", None, None, "ck_returns_received_at"),
        ("in_transit", "now()", "pending", "ck_returns_received_at"),
        ("received", "now()", None, "ck_returns_inspection_after_receipt"),
        ("in_transit", None, "passed", "ck_returns_inspection_after_receipt"),
        ("lost", None, None, "ck_returns_status"),
    ],
)
def test_invalid_returns_are_rejected(
    rows: Connection,
    status: str,
    received_at: str | None,
    inspection_status: str | None,
    constraint: str,
) -> None:
    sql = (
        "INSERT INTO support.returns (id, order_id, status, requested_at, received_at, "
        f"inspection_status) VALUES ('ret_x', 'ord_a', :status, now(), {received_at or 'NULL'}, "
        ":inspection_status)"
    )
    params = {"status": status, "inspection_status": inspection_status}
    assert rejected(rows, sql, params) == constraint


def test_one_return_per_order(rows: Connection) -> None:
    sql = (
        "INSERT INTO support.returns (id, order_id, status, requested_at) "
        "VALUES (:id, 'ord_a', 'requested', now())"
    )
    rows.execute(text(sql), {"id": "ret_1"})
    assert rejected(rows, sql, {"id": "ret_2"}) == "uq_returns_order_id"


# --- tickets, runs, policies --------------------------------------------------------------


def test_ticket_cannot_reference_another_customers_order(rows: Connection) -> None:
    sql = (
        "INSERT INTO support.tickets (id, customer_id, order_id, status, customer_message) "
        "VALUES ('tkt_x', 'cus_a', 'ord_b', 'open', 'Where is my order?')"
    )
    assert rejected(rows, sql) == "fk_tickets_order_owner"


def test_only_one_active_run_per_ticket(rows: Connection) -> None:
    sql = (
        "INSERT INTO support.workflow_runs (id, ticket_id, thread_id, status) "
        "VALUES ('run_2', 'tkt_1', 'th_2', 'waiting_for_approval')"
    )
    assert rejected(rows, sql) == "uq_workflow_runs_one_active_per_ticket"


def test_policy_versions_cannot_overlap(rows: Connection) -> None:
    sql = (
        "INSERT INTO support.policy_documents (id, policy_key, version, title, effective_from, "
        "effective_to, source_uri, content_hash) "
        "VALUES (:id, 'refund_policy', :version, 'Refund', :start, :end, 'uri', 'hash')"
    )
    rows.execute(text(sql), {"id": "rp_v0", "version": "v0", "start": "2025-01-01", "end": None})
    params = {"id": "rp_v1", "version": "v1", "start": "2026-01-01", "end": None}
    assert rejected(rows, sql, params) == "ex_policy_documents_no_overlap"


def test_embedding_must_have_384_dimensions(rows: Connection) -> None:
    rows.execute(
        text(
            "INSERT INTO support.policy_documents (id, policy_key, version, title, "
            "effective_from, source_uri, content_hash) "
            "VALUES ('rp_v1', 'refund_policy', 'v1', 'Refund', '2026-01-01', 'uri', 'hash')"
        )
    )
    sql = (
        "INSERT INTO support.policy_chunks "
        "(id, document_id, section_number, section_title, content, embedding) "
        "VALUES ('rp_v1:section_1', 'rp_v1', 1, 'Scope', 'text', "
        "array_fill(0.1::real, ARRAY[:dimensions])::vector)"
    )
    assert "expected 384 dimensions" in rejected(rows, sql, {"dimensions": 3})
    rows.execute(text(sql), {"dimensions": 384})


# --- proposals and approvals --------------------------------------------------------------


def test_proposal_payload_is_immutable(rows: Connection) -> None:
    sql = "UPDATE support.action_proposals SET payload = '{\"amount_minor\": 1}' WHERE id = 'prp_1'"
    assert "is immutable" in rejected(rows, sql)
    rows.execute(text("UPDATE support.action_proposals SET status = 'executed' WHERE id = 'prp_1'"))


def test_proposal_cannot_target_another_customers_order(rows: Connection) -> None:
    sql = (
        "INSERT INTO support.action_proposals "
        "(id, run_id, order_id, customer_id, action_type, payload, payload_hash, status) "
        f"VALUES ('prp_x', 'run_1', 'ord_b', 'cus_a', 'refund', '{{}}', '{'c' * 64}', "
        "'pending_approval')"
    )
    assert rejected(rows, sql) == "fk_action_proposals_order_owner"


def test_approval_is_bound_to_the_proposal_hash(rows: Connection) -> None:
    assert rejected(rows, INSERT_APPROVAL, approval(hash=HASH_2)) == "fk_approvals_proposal_hash"


def test_only_one_pending_approval_per_proposal(rows: Connection) -> None:
    rows.execute(text(INSERT_APPROVAL), approval(id="apr_p1"))
    assert (
        rejected(rows, INSERT_APPROVAL, approval(id="apr_p2"))
        == "uq_approvals_one_pending_per_proposal"
    )


def test_expired_approval_can_be_replaced(rows: Connection) -> None:
    rows.execute(text(INSERT_APPROVAL), approval(id="apr_p1", expires_in_minutes=-1))
    rows.execute(
        text(
            "UPDATE support.approvals SET decision = 'expired', decided_at = now() "
            "WHERE id = 'apr_p1'"
        )
    )
    rows.execute(text(INSERT_APPROVAL), approval(id="apr_p2"))


def test_reviewer_cannot_decide_after_expiry(rows: Connection) -> None:
    params = approval(
        reviewer_id="rev_1", reviewer_role="reviewer", decision="approved", expires_in_minutes=-1
    )
    assert rejected(rows, INSERT_APPROVAL, params) == "ck_approvals_decided_before_expiry"


def test_high_value_refund_needs_a_senior_reviewer(rows: Connection) -> None:
    params = approval(
        required_role="senior_reviewer",
        reviewer_id="rev_1",
        reviewer_role="reviewer",
        decision="approved",
    )
    assert rejected(rows, INSERT_APPROVAL, params) == "ck_approvals_senior_reviewer"
    rows.execute(text(INSERT_APPROVAL), params | {"reviewer_role": "senior_reviewer"})


@pytest.mark.parametrize(
    ("reviewer_id", "reviewer_role", "decision"),
    [("rev_1", None, "approved"), ("rev_1", "reviewer", "expired"), (None, None, "rejected")],
)
def test_reviewer_fields_match_the_decision(
    rows: Connection, reviewer_id: str | None, reviewer_role: str | None, decision: str
) -> None:
    params = approval(reviewer_id=reviewer_id, reviewer_role=reviewer_role, decision=decision)
    assert rejected(rows, INSERT_APPROVAL, params) == "ck_approvals_reviewer_fields"


# --- refund operations --------------------------------------------------------------------


def test_refund_needs_an_approved_approval(rows: Connection) -> None:
    rows.execute(text(INSERT_APPROVAL), approval(id="apr_pending"))
    assert "is not approved" in rejected(rows, INSERT_REFUND, refund(approval_id="apr_pending"))


def test_refund_uses_the_approval_of_its_own_proposal(rows: Connection) -> None:
    params = refund(approval_id="apr_2")
    assert rejected(rows, INSERT_REFUND, params) == "fk_refund_operations_approval_proposal"


def test_refund_is_for_the_proposals_order(rows: Connection) -> None:
    params = refund(order_id="ord_b")
    assert rejected(rows, INSERT_REFUND, params) == "fk_refund_operations_proposal_order"


def test_idempotency_key_is_unique(rows: Connection) -> None:
    rows.execute(text(INSERT_REFUND), refund())
    params = refund(id="rop_2", proposal_id="prp_2", approval_id="apr_2")
    assert rejected(rows, INSERT_REFUND, params) == "uq_refund_operations_idempotency_key"


def test_succeeded_refund_needs_a_provider_reference(rows: Connection) -> None:
    params = refund(status="succeeded")
    assert rejected(rows, INSERT_REFUND, params) == "ck_refund_operations_succeeded_reference"


def test_refunds_cannot_exceed_the_paid_amount(rows: Connection) -> None:
    rows.execute(text(INSERT_REFUND), refund(amount=249900))
    params = refund(id="rop_2", proposal_id="prp_2", approval_id="apr_2", key="refund:prp_2")
    assert "exceeds refundable balance" in rejected(rows, INSERT_REFUND, params)


def test_failed_refund_releases_its_reservation(rows: Connection) -> None:
    rows.execute(text(INSERT_REFUND), refund(amount=249900, status="failed"))
    second = refund(
        id="rop_2", proposal_id="prp_2", approval_id="apr_2", key="refund:prp_2", amount=100000
    )
    rows.execute(text(INSERT_REFUND), second)
    # Reviving the failed operation would now over-commit the order.
    sql = "UPDATE support.refund_operations SET status = 'reserved' WHERE id = 'rop_1'"
    assert "exceeds refundable balance" in rejected(rows, sql)


def test_concurrent_refunds_cannot_over_reserve(engine: Engine) -> None:
    """Two sessions reserving ₹2,000 and ₹1,000 on a ₹2,499 order: exactly one succeeds."""
    with engine.begin() as setup:
        insert_base_rows(setup)
    first_reserved = threading.Event()

    def first_session() -> None:
        with engine.begin() as connection:
            connection.execute(text(INSERT_REFUND), refund(amount=200000))
            first_reserved.set()
            time.sleep(1)

    try:
        thread = threading.Thread(target=first_session)
        thread.start()
        assert first_reserved.wait(timeout=10)
        started = time.monotonic()
        second = refund(
            id="rop_2", proposal_id="prp_2", approval_id="apr_2", key="refund:prp_2", amount=100000
        )
        with pytest.raises(DBAPIError, match="exceeds refundable balance"), engine.begin() as c:
            c.execute(text(INSERT_REFUND), second)
        # The second session waited on the order row lock until the first one committed.
        assert time.monotonic() - started > 0.5
        thread.join()

        with engine.connect() as connection:
            amounts = connection.execute(
                text("SELECT amount_minor FROM support.refund_operations")
            ).scalars()
            assert list(amounts) == [200000]
    finally:
        with engine.begin() as cleanup:
            # TRUNCATE skips row-level triggers, so the append-only audit table is cleared too.
            cleanup.execute(
                text(
                    f"TRUNCATE {SCHEMA}.customers, {SCHEMA}.policy_documents, "
                    f"{SCHEMA}.audit_events CASCADE"
                )
            )


# --- audit trail --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE support.audit_events SET event_type = 'edited'",
        "DELETE FROM support.audit_events",
    ],
)
def test_audit_trail_is_append_only(rows: Connection, sql: str) -> None:
    rows.execute(
        text(
            "INSERT INTO support.audit_events (run_id, actor, event_type) "
            "VALUES ('run_1', 'agent:investigation', 'tool_call')"
        )
    )
    assert "append-only" in rejected(rows, sql)


def test_audit_actor_names_its_kind(rows: Connection) -> None:
    sql = (
        "INSERT INTO support.audit_events (run_id, actor, event_type) "
        "VALUES ('run_1', 'model', 'tool_call')"
    )
    assert rejected(rows, sql) == "ck_audit_events_actor"
