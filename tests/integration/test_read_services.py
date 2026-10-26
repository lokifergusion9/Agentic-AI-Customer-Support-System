"""Read services: ownership is checked in the service before any provider is asked (W1.12)."""

from collections.abc import Iterator
from datetime import date
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.services.errors import ServiceFailure
from app.services.payments import get_payment_status, get_refund_status
from app.services.providers import ProviderClient
from app.services.returns import get_return_status
from app.services.shipping import get_shipment_status
from app.services.tickets import get_ticket
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from scripts.seed_data import at, push_mock_payments, push_mock_shipping, seed

TODAY = date(2026, 10, 25)
ALICE = AuthenticatedCustomer("cus_alice", "auth|alice")
BOB = AuthenticatedCustomer("cus_bob", "auth|bob")
CHITRA = AuthenticatedCustomer("cus_chitra", "auth|chitra")


class CountingProvider(ProviderClient):
    """A provider client that records every call, to show ownership is checked first."""

    def __init__(self, name: str, http: httpx.Client) -> None:
        super().__init__(name, http)
        self.calls: list[str] = []

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        self.calls.append(f"{method} {path}")
        return super().request(method, path, **kwargs)


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    seed(conn, TODAY)
    with Session(bind=conn) as session:
        yield session


@pytest.fixture
def payments() -> Iterator[CountingProvider]:
    headers = {"X-API-Key": payments_settings.mock_payments_api_key}
    with TestClient(payments_app, headers=headers) as http:
        push_mock_payments(http, TODAY)
        yield CountingProvider("payment", http)


@pytest.fixture
def shipping() -> Iterator[CountingProvider]:
    headers = {"X-API-Key": shipping_settings.mock_shipping_api_key}
    with TestClient(shipping_app, headers=headers) as http:
        push_mock_shipping(http, TODAY)
        yield CountingProvider("shipping", http)


def unreachable(name: str) -> ProviderClient:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    return ProviderClient(
        name, httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://x")
    )


def failure(call) -> ServiceFailure:
    with pytest.raises(ServiceFailure) as error:
        call()
    return error.value


# --- get_return_status ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("customer", "order_id", "status", "inspection"),
    [
        (ALICE, "ord_1001", "received", "passed"),
        (ALICE, "ord_1002", "closed", "passed"),
        (BOB, "ord_1004", "in_transit", None),
        (CHITRA, "ord_1005", "received", "failed"),
        (CHITRA, "ord_1006", "received", "pending"),
    ],
)
def test_return_states(
    db: Session, customer: AuthenticatedCustomer, order_id: str, status: str, inspection: str | None
) -> None:
    result = get_return_status(db, customer, order_id)
    assert (result.order_id, result.status, result.inspection_status) == (
        order_id,
        status,
        inspection,
    )
    assert (result.received_at is None) == (status == "in_transit")


def test_return_dates(db: Session) -> None:
    result = get_return_status(db, ALICE, "ord_1001")
    assert result.return_id == "ret_1001"
    assert (result.requested_at, result.received_at) == (
        at(date(2026, 10, 18)),
        at(date(2026, 10, 22)),
    )


def test_order_without_a_return(db: Session) -> None:
    result = failure(lambda: get_return_status(db, ALICE, "ord_1008"))
    assert (result.error_code, result.retryable) == ("RETURN_NOT_FOUND", False)


@pytest.mark.parametrize("order_id", ["ord_1004", "ord_9999"], ids=["bobs", "missing"])
def test_return_of_someone_elses_or_missing_order(db: Session, order_id: str) -> None:
    # ORDER_NOT_FOUND, not RETURN_NOT_FOUND: nothing about Bob's order may leak.
    assert failure(lambda: get_return_status(db, ALICE, order_id)).error_code == "ORDER_NOT_FOUND"


# --- get_ticket -------------------------------------------------------------------------


def test_own_ticket(db: Session) -> None:
    ticket = get_ticket(db, BOB, "tkt_1003")
    assert (ticket.order_id, ticket.category, ticket.status) == (
        "ord_1003",
        "refund_request",
        "open",
    )
    assert (ticket.summary, ticket.external_ticket_ref) == (
        "Refund submitted, awaiting provider.",
        "ext_1003",
    )
    assert ticket.customer_message.startswith("I returned my item")


@pytest.mark.parametrize("ticket_id", ["tkt_1003", "tkt_nope"], ids=["bobs", "missing"])
def test_someone_elses_or_missing_ticket(db: Session, ticket_id: str) -> None:
    result = failure(lambda: get_ticket(db, ALICE, ticket_id))
    assert (result.error_code, result.message) == (
        "TICKET_NOT_FOUND",
        "No ticket with this ID was found.",
    )


# --- get_shipment_status ----------------------------------------------------------------


def test_delivered_shipment(db: Session, shipping: CountingProvider) -> None:
    result = get_shipment_status(db, shipping, ALICE, "ord_1001")
    assert (result.tracking_id, result.status) == ("trk_1001", "delivered")
    assert result.delivered_at == result.last_updated_at == result.events[-1].at
    assert shipping.calls == ["GET /shipments"]


def test_delayed_shipment(db: Session, shipping: CountingProvider) -> None:
    result = get_shipment_status(db, shipping, CHITRA, "ord_1010")
    assert (result.status, result.delivered_at) == ("in_transit", None)
    assert result.estimated_delivery == date(2026, 10, 14)
    assert result.last_updated_at.date() == date(2026, 10, 13)


def test_exception_event_is_reported(db: Session, shipping: CountingProvider) -> None:
    shipping.http.post(
        "/admin/shipments/trk_1009/events", json={"status": "exception", "location": "Pune hub"}
    )
    assert get_shipment_status(db, shipping, BOB, "ord_1009").status == "exception"


@pytest.mark.parametrize("order_id", ["ord_1009", "ord_9999"], ids=["bobs", "missing"])
def test_shipment_ownership_is_checked_before_the_provider(
    db: Session, shipping: CountingProvider, order_id: str
) -> None:
    assert failure(lambda: get_shipment_status(db, shipping, ALICE, order_id)).error_code == (
        "ORDER_NOT_FOUND"
    )
    assert shipping.calls == []


def test_order_the_courier_does_not_know(db: Session, shipping: CountingProvider) -> None:
    shipping.http.post("/admin/reset", json={"shipments": []})
    result = failure(lambda: get_shipment_status(db, shipping, ALICE, "ord_1001"))
    assert (result.error_code, result.retryable) == ("SHIPMENT_NOT_FOUND", False)


def test_shipping_errors_are_structured(db: Session, shipping: CountingProvider) -> None:
    shipping.http.post("/admin/faults", json={"mode": "error"})
    result = failure(lambda: get_shipment_status(db, shipping, ALICE, "ord_1001"))
    assert (result.error_code, result.retryable) == ("PROVIDER_UNAVAILABLE", True)
    unavailable = failure(
        lambda: get_shipment_status(db, unreachable("shipping"), ALICE, "ord_1001")
    )
    assert (unavailable.error_code, unavailable.retryable) == ("PROVIDER_UNAVAILABLE", True)


# --- get_payment_status -----------------------------------------------------------------


def test_payment_without_refunds(db: Session, payments: CountingProvider) -> None:
    result = get_payment_status(db, payments, ALICE, "ord_1001")
    assert (result.payment_id, result.status, result.currency) == ("pay_1001", "captured", "INR")
    assert (result.amount_minor, result.refundable_minor, result.refunds) == (249_900, 249_900, [])


@pytest.mark.parametrize(
    ("customer", "order_id", "refund_status"),
    [(ALICE, "ord_1002", "succeeded"), (BOB, "ord_1003", "pending")],
)
def test_payment_with_an_earlier_refund(
    db: Session,
    payments: CountingProvider,
    customer: AuthenticatedCustomer,
    order_id: str,
    refund_status: str,
) -> None:
    result = get_payment_status(db, payments, customer, order_id)
    assert result.refundable_minor == 0
    assert [(r.provider_reference, r.status) for r in result.refunds] == [
        (f"rf_{order_id.removeprefix('ord_')}", refund_status)
    ]


@pytest.mark.parametrize("order_id", ["ord_1003", "ord_9999"], ids=["bobs", "missing"])
def test_payment_ownership_is_checked_before_the_provider(
    db: Session, payments: CountingProvider, order_id: str
) -> None:
    assert failure(lambda: get_payment_status(db, payments, ALICE, order_id)).error_code == (
        "ORDER_NOT_FOUND"
    )
    assert payments.calls == []


def test_payment_unknown_to_the_provider(db: Session, payments: CountingProvider) -> None:
    payments.http.post("/admin/reset", json={"payments": []})
    assert failure(lambda: get_payment_status(db, payments, ALICE, "ord_1001")).error_code == (
        "PAYMENT_NOT_FOUND"
    )


def test_payment_provider_unreachable(db: Session) -> None:
    result = failure(lambda: get_payment_status(db, unreachable("payment"), ALICE, "ord_1001"))
    assert (result.error_code, result.retryable) == ("PROVIDER_UNAVAILABLE", True)


# --- get_refund_status ------------------------------------------------------------------


def test_succeeded_refund_is_confirmed_by_the_provider(
    db: Session, payments: CountingProvider
) -> None:
    result = get_refund_status(db, payments, ALICE, "rop_1002")
    assert (result.order_id, result.amount_minor) == ("ord_1002", 399_900)
    assert (result.local_status, result.provider_status, result.matches) == (
        "succeeded",
        "succeeded",
        True,
    )
    assert (result.provider_reference, result.provider_error) == ("rf_1002", None)
    assert payments.calls == ["GET /refunds"]


def test_submitted_refund_is_pending_at_the_provider(
    db: Session, payments: CountingProvider
) -> None:
    result = get_refund_status(db, payments, BOB, "rop_1003")
    assert (result.local_status, result.provider_status, result.matches) == (
        "submitted",
        "pending",
        True,
    )


def test_provider_settling_the_refund_makes_it_disagree(
    db: Session, payments: CountingProvider
) -> None:
    payments.http.post("/admin/refunds/rf_1003/settle", json={"status": "succeeded"})
    result = get_refund_status(db, payments, BOB, "rop_1003")
    assert (result.local_status, result.provider_status, result.matches) == (
        "submitted",
        "succeeded",
        False,
    )


def test_provider_without_a_record_disagrees_with_a_success(
    db: Session, payments: CountingProvider
) -> None:
    payments.http.post("/admin/reset", json={"payments": []})
    result = get_refund_status(db, payments, ALICE, "rop_1002")
    assert (result.provider_status, result.matches) == ("no_record", False)
    assert result.provider_reference == "rf_1002"  # the local reference is kept


def test_unknown_local_status_never_matches(db: Session, payments: CountingProvider) -> None:
    db.execute(
        text("UPDATE support.refund_operations SET status = 'unknown' WHERE id = 'rop_1003'")
    )
    result = get_refund_status(db, payments, BOB, "rop_1003")
    assert (result.local_status, result.provider_status, result.matches) == (
        "unknown",
        "pending",
        False,
    )


def test_provider_reference_is_filled_from_the_provider(
    db: Session, payments: CountingProvider
) -> None:
    db.execute(
        text(
            "UPDATE support.refund_operations SET status = 'unknown', provider_reference = NULL "
            "WHERE id = 'rop_1003'"
        )
    )
    assert get_refund_status(db, payments, BOB, "rop_1003").provider_reference == "rf_1003"


def test_unreachable_provider_still_returns_the_local_record(db: Session) -> None:
    result = get_refund_status(db, unreachable("payment"), ALICE, "rop_1002")
    assert (result.local_status, result.provider_reference) == ("succeeded", "rf_1002")
    assert (result.provider_status, result.provider_error, result.matches) == (
        None,
        "PROVIDER_UNAVAILABLE",
        None,
    )


@pytest.mark.parametrize("operation_id", ["rop_1003", "rop_nope"], ids=["bobs", "missing"])
def test_someone_elses_or_missing_refund(
    db: Session, payments: CountingProvider, operation_id: str
) -> None:
    result = failure(lambda: get_refund_status(db, payments, ALICE, operation_id))
    assert (result.error_code, result.message) == (
        "REFUND_NOT_FOUND",
        "No refund with this ID was found.",
    )
    assert payments.calls == []
