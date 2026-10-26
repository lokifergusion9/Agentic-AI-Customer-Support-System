"""The seed data loads cleanly and each scenario has the facts its eval cases rely on (W1.2)."""

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import Connection, text

from scripts.seed_data import (
    business_days_before,
    mock_payments_state,
    mock_shipping_state,
    scenarios,
    seed,
)

TODAY = date(2026, 10, 25)  # A Sunday, so business-day offsets skip a weekend.


@pytest.fixture
def seeded(conn: Connection) -> Connection:
    seed(conn, TODAY)
    return conn


def one(conn: Connection, sql: str, **params: object):
    return conn.execute(text(sql), params).one()


def test_seed_loads_every_scenario(seeded: Connection) -> None:
    counts = one(
        seeded,
        "SELECT (SELECT count(*) FROM support.customers) AS customers, "
        "(SELECT count(*) FROM support.orders) AS orders",
    )
    assert (counts.customers, counts.orders) == (4, 15)


def test_paid_amount_is_items_plus_shipping(seeded: Connection) -> None:
    mismatches = seeded.execute(
        text(
            "SELECT o.id FROM support.orders o JOIN support.order_items i ON i.order_id = o.id "
            "GROUP BY o.id HAVING o.paid_amount_minor <> sum(i.paid_amount_minor) "
            "+ o.shipping_amount_minor"
        )
    ).all()
    assert mismatches == []


def test_reference_case_is_eligible_on_paper(seeded: Connection) -> None:
    row = one(
        seeded,
        "SELECT o.paid_amount_minor, o.delivered_at::date AS delivered, r.requested_at::date "
        "AS requested, r.status, r.inspection_status, i.product_name "
        "FROM support.orders o JOIN support.returns r ON r.order_id = o.id "
        "JOIN support.order_items i ON i.order_id = o.id WHERE o.id = 'ord_1001'",
    )
    assert row.product_name == "Wireless Headphones"
    assert row.paid_amount_minor == 249900
    assert (row.status, row.inspection_status) == ("received", "passed")
    assert row.requested - row.delivered <= timedelta(days=10)


def test_past_refunds_have_a_full_approval_chain(seeded: Connection) -> None:
    rows = seeded.execute(
        text(
            "SELECT r.order_id, r.status, r.amount_minor, o.paid_amount_minor, a.decision, "
            "a.approved_payload_hash = p.payload_hash AS hash_matches "
            "FROM support.refund_operations r JOIN support.approvals a ON a.id = r.approval_id "
            "JOIN support.action_proposals p ON p.id = r.proposal_id "
            "JOIN support.orders o ON o.id = r.order_id ORDER BY r.order_id"
        )
    ).all()
    assert [(row.order_id, row.status) for row in rows] == [
        ("ord_1002", "succeeded"),
        ("ord_1003", "submitted"),
    ]
    for row in rows:
        assert row.amount_minor == row.paid_amount_minor
        assert row.decision == "approved"
        assert row.hash_matches


def test_policy_v0_order_keeps_its_2025_dates(conn: Connection) -> None:
    for today in (TODAY, date(2030, 1, 1)):
        seed(conn, today)
        row = one(
            conn,
            "SELECT o.delivered_at::date AS delivered, r.requested_at::date AS requested "
            "FROM support.orders o JOIN support.returns r ON r.order_id = o.id "
            "WHERE o.id = 'ord_1013'",
        )
        assert (row.delivered, row.requested) == (date(2025, 6, 10), date(2025, 6, 18))


@pytest.mark.parametrize(("order_id", "days_late"), [("ord_1008", 2), ("ord_1009", 5)])
def test_delays_are_counted_in_business_days(order_id: str, days_late: int) -> None:
    shipment = next(s for s in mock_shipping_state(TODAY) if s["order_id"] == order_id)
    estimate = date.fromisoformat(shipment["estimated_delivery"])
    assert shipment["status"] == "in_transit"
    assert estimate == business_days_before(TODAY, days_late)
    assert estimate.weekday() < 5


def test_business_days_skip_weekends() -> None:
    assert business_days_before(date(2026, 10, 26), 1) == date(2026, 10, 23)  # Mon -> Fri
    assert business_days_before(date(2026, 10, 25), 8) == date(2026, 10, 14)  # Sun -> Wed


def test_high_value_order_is_above_the_senior_threshold() -> None:
    laptop = next(s for s in scenarios(TODAY) if s.order_id == "ord_1014")
    assert laptop.paid_amount_minor > 5000000


def test_mock_payments_match_the_database(seeded: Connection) -> None:
    paid = dict(seeded.execute(text("SELECT id, paid_amount_minor FROM support.orders")).all())
    payments = mock_payments_state(TODAY)
    assert {p["order_id"]: p["amount_minor"] for p in payments} == paid
    refunded = {p["order_id"]: p["refunds"][0]["status"] for p in payments if p["refunds"]}
    assert refunded == {"ord_1002": "succeeded", "ord_1003": "pending"}


def test_mock_shipping_matches_the_database(seeded: Connection) -> None:
    orders = {
        row.id: row
        for row in seeded.execute(text("SELECT id, status, delivered_at FROM support.orders"))
    }
    shipments = {s["order_id"]: s for s in mock_shipping_state(TODAY)}
    assert shipments.keys() == orders.keys()
    for order_id, shipment in shipments.items():
        order = orders[order_id]
        if order.status == "delivered":
            assert shipment["status"] == "delivered"
            assert datetime.fromisoformat(shipment["delivered_at"]) == order.delivered_at
        else:
            assert (order.status, shipment["status"]) == ("shipped", "in_transit")
            assert shipment["delivered_at"] is None and order.delivered_at is None


def test_reseeding_replaces_business_data_and_keeps_policies(seeded: Connection) -> None:
    seeded.execute(
        text(
            "INSERT INTO support.policy_documents (id, policy_key, version, title, "
            "effective_from, source_uri, content_hash) "
            "VALUES ('refund_policy_v1', 'refund_policy', 'v1', 'Refund', '2026-01-01', 'u', 'h')"
        )
    )
    before = seeded.execute(text("SELECT * FROM support.orders ORDER BY id")).all()

    seed(seeded, TODAY)

    after = seeded.execute(text("SELECT * FROM support.orders ORDER BY id")).all()
    # created_at is set by the database at insert time, so compare everything else.
    assert [row[:-1] for row in after] == [row[:-1] for row in before]
    assert one(seeded, "SELECT count(*) AS n FROM support.policy_documents").n == 1
