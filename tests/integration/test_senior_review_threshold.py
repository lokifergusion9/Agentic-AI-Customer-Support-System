"""The senior-review threshold applies to the refund amount, checked on seeded orders (W1.6)."""

from datetime import date

import pytest
from sqlalchemy import Connection, text

from app.auth.reviewer import required_reviewer_role
from scripts.seed_data import seed


@pytest.fixture
def seeded(conn: Connection) -> Connection:
    seed(conn, date(2026, 10, 25))
    return conn


def paid(conn: Connection, order_id: str) -> int:
    return conn.execute(
        text("SELECT paid_amount_minor FROM support.orders WHERE id = :id"), {"id": order_id}
    ).scalar_one()


def item_amount(conn: Connection, order_id: str, product_id: str) -> int:
    return conn.execute(
        text(
            "SELECT paid_amount_minor FROM support.order_items "
            "WHERE order_id = :order AND product_id = :product"
        ),
        {"order": order_id, "product": product_id},
    ).scalar_one()


def test_high_value_order_needs_a_senior_reviewer(seeded: Connection) -> None:
    assert required_reviewer_role(paid(seeded, "ord_1014")) == "senior_reviewer"


def test_eligible_headphones_need_a_standard_reviewer(seeded: Connection) -> None:
    assert required_reviewer_role(paid(seeded, "ord_1001")) == "reviewer"


def test_threshold_uses_the_refund_amount_not_the_order_total(seeded: Connection) -> None:
    # ord_1015 totals above ₹50,000, but refunding only the lens stays below it.
    assert required_reviewer_role(paid(seeded, "ord_1015")) == "senior_reviewer"
    assert required_reviewer_role(item_amount(seeded, "ord_1015", "prd_lens")) == "reviewer"
