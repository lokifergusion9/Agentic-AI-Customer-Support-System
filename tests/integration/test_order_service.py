"""Order read service: ownership is checked in the service, results are typed (W1.11)."""

from collections.abc import Iterator
from datetime import date

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.services.errors import ServiceFailure
from app.services.orders import get_order, list_orders
from scripts.seed_data import at, scenarios, seed

TODAY = date(2026, 10, 25)
ALICE = AuthenticatedCustomer("cus_alice", "auth|alice")
BOB = AuthenticatedCustomer("cus_bob", "auth|bob")
CHITRA = AuthenticatedCustomer("cus_chitra", "auth|chitra")
SCENARIOS = {s.order_id: s for s in scenarios(TODAY)}


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    seed(conn, TODAY)
    with Session(bind=conn) as session:
        yield session


def failure(call) -> dict:
    with pytest.raises(ServiceFailure) as error:
        call()
    return error.value.to_response().model_dump()


# --- get_order --------------------------------------------------------------------------


def test_own_order(db: Session) -> None:
    order = get_order(db, ALICE, "ord_1001")
    assert (order.id, order.status, order.currency) == ("ord_1001", "delivered", "INR")
    assert (order.paid_amount_minor, order.shipping_amount_minor) == (249_900, 0)
    assert order.placed_at == at(SCENARIOS["ord_1001"].placed)
    assert order.delivered_at == at(SCENARIOS["ord_1001"].delivered)
    assert [
        (i.product_id, i.product_name, i.quantity, i.paid_amount_minor) for i in order.items
    ] == [("prd_headphones", "Wireless Headphones", 1, 249_900)]


def test_multi_item_order(db: Session) -> None:
    order = get_order(db, CHITRA, "ord_1015")
    assert [i.product_id for i in order.items] == ["prd_camera", "prd_lens", "prd_camerabag"]
    items_total = sum(i.paid_amount_minor for i in order.items)
    assert items_total + order.shipping_amount_minor == order.paid_amount_minor == 6_099_700


def test_paid_amount_includes_shipping(db: Session) -> None:
    order = get_order(db, AuthenticatedCustomer("cus_dev", "auth|dev"), "ord_1007")
    assert order.shipping_amount_minor == 9_900
    assert sum(i.paid_amount_minor for i in order.items) + 9_900 == order.paid_amount_minor


def test_shipped_order_is_not_delivered_yet(db: Session) -> None:
    order = get_order(db, CHITRA, "ord_1010")
    assert (order.status, order.delivered_at) == ("shipped", None)


def test_2025_order_keeps_its_dates(db: Session) -> None:
    order = get_order(db, ALICE, "ord_1013")
    assert order.placed_at.year == order.delivered_at.year == 2025


@pytest.mark.parametrize("order_id", ["ord_1014", "ord_9999", ""], ids=["bobs", "missing", "empty"])
def test_other_or_missing_order_fails_the_same_way(db: Session, order_id: str) -> None:
    assert failure(lambda: get_order(db, ALICE, order_id)) == {
        "success": False,
        "error_code": "ORDER_NOT_FOUND",
        "retryable": False,
        "message": "No order with this ID was found.",
    }


def test_owner_can_read_the_order_another_customer_cannot(db: Session) -> None:
    assert get_order(db, BOB, "ord_1014").paid_amount_minor == 8_999_900
    with pytest.raises(ServiceFailure):
        get_order(db, ALICE, "ord_1014")


# --- list_orders ------------------------------------------------------------------------


def expected_ids(customer: AuthenticatedCustomer) -> list[str]:
    own = [s for s in SCENARIOS.values() if s.customer_id == customer.customer_id]
    return [s.order_id for s in sorted(own, key=lambda s: (-s.placed.toordinal(), s.order_id))]


@pytest.mark.parametrize("customer", [ALICE, BOB, CHITRA], ids=lambda c: c.customer_id)
def test_lists_only_own_orders_newest_first(db: Session, customer: AuthenticatedCustomer) -> None:
    orders = list_orders(db, customer)
    assert [o.id for o in orders] == expected_ids(customer)
    placed = [o.placed_at for o in orders]
    assert placed == sorted(placed, reverse=True)


def test_list_has_product_names_and_amounts(db: Session) -> None:
    summary = next(o for o in list_orders(db, CHITRA) if o.id == "ord_1015")
    assert summary.product_names == ["Mirrorless Camera", "50mm Lens", "Camera Bag"]
    assert (summary.status, summary.paid_amount_minor, summary.currency) == (
        "delivered",
        6_099_700,
        "INR",
    )


def test_list_limit(db: Session) -> None:
    assert [o.id for o in list_orders(db, ALICE, limit=2)] == expected_ids(ALICE)[:2]


def test_customer_without_orders_gets_an_empty_list(db: Session) -> None:
    db.execute(
        text(
            "INSERT INTO support.customers (id, external_auth_subject, name, email) "
            "VALUES ('cus_new', 'auth|new', 'New Customer', 'new@example.test')"
        )
    )
    assert list_orders(db, AuthenticatedCustomer("cus_new", "auth|new")) == []
