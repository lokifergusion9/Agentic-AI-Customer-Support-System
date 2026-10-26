"""Refund eligibility on the seeded scenarios, with facts from the database and provider (W1.14)."""

from collections.abc import Iterator
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.services.eligibility import check_refund_eligibility
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from scripts.seed_data import push_mock_payments, scenarios, seed

TODAY = date(2026, 10, 25)
OWNERS = {s.order_id: s.customer_id for s in scenarios(TODAY)}


CASES = [
    ("ord_1001", "eligible", None, "v1"),
    ("ord_1002", "blocked", "ALREADY_REFUNDED", "v1"),
    ("ord_1003", "already_processing", "REFUND_PENDING", "v1"),
    ("ord_1004", "blocked", "RETURN_NOT_RECEIVED", "v1"),
    ("ord_1005", "blocked", "INSPECTION_FAILED", "v1"),
    ("ord_1006", "blocked", "INSPECTION_PENDING", "v1"),
    # Damaged items fall under the Damaged Products Policy, not this one (W2).
    ("ord_1007", "blocked", "RETURN_NOT_REQUESTED", "v1"),
    ("ord_1008", "blocked", "ORDER_NOT_DELIVERED", None),
    ("ord_1009", "blocked", "ORDER_NOT_DELIVERED", None),
    ("ord_1010", "blocked", "ORDER_NOT_DELIVERED", None),
    ("ord_1011", "blocked", "RETURN_NOT_REQUESTED", "v1"),
    ("ord_1012", "needs_review", "OUTSIDE_RETURN_WINDOW", "v1"),
    ("ord_1013", "needs_review", "OUTSIDE_RETURN_WINDOW", "v0"),
    ("ord_1014", "eligible", None, "v1"),
    # Partial returns of multi-item orders are out of scope (section 8), routed in W2.
    ("ord_1015", "blocked", "RETURN_NOT_REQUESTED", "v1"),
]


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    seed(conn, TODAY)
    with Session(bind=conn) as session:
        yield session


@pytest.fixture
def payments() -> Iterator[ProviderClient]:
    headers = {"X-API-Key": payments_settings.mock_payments_api_key}
    with TestClient(payments_app, headers=headers) as http:
        push_mock_payments(http, TODAY)
        yield ProviderClient("payment", http)


def owner(order_id: str) -> AuthenticatedCustomer:
    customer_id = OWNERS[order_id]
    return AuthenticatedCustomer(customer_id, f"auth|{customer_id.removeprefix('cus_')}")


def check(db: Session, payments: ProviderClient, order_id: str, **kwargs: object):
    return check_refund_eligibility(db, payments, owner(order_id), order_id, today=TODAY, **kwargs)


@pytest.mark.parametrize(("order_id", "outcome", "reason", "version"), CASES)
def test_every_seeded_scenario(
    db: Session,
    payments: ProviderClient,
    order_id: str,
    outcome: str,
    reason: str | None,
    version: str | None,
) -> None:
    result = check(db, payments, order_id)
    assert (result.order_id, result.outcome, result.reason, result.policy_version) == (
        order_id,
        outcome,
        reason,
        version,
    )


def test_every_scenario_is_covered() -> None:
    assert {case[0] for case in CASES} == set(OWNERS)


def test_reference_refund(db: Session, payments: ProviderClient) -> None:
    result = check(db, payments, "ord_1001")
    assert (result.amount_minor, result.max_refund_minor, result.reviewer_role) == (
        249_900,
        249_900,
        "reviewer",
    )
    assert result.policy_refs == [
        "refund_policy_v1:section_3",
        "refund_policy_v1:section_4",
        "refund_policy_v1:section_5",
    ]


def test_high_value_refund_needs_a_senior_reviewer(db: Session, payments: ProviderClient) -> None:
    result = check(db, payments, "ord_1014")
    assert (result.amount_minor, result.reviewer_role) == (8_999_900, "senior_reviewer")
    assert "human_review_exceptions_v1:section_3" in result.policy_refs


def test_requested_amount(db: Session, payments: ProviderClient) -> None:
    partial = check(db, payments, "ord_1001", amount_minor=100_000)
    assert (partial.outcome, partial.amount_minor) == ("eligible", 100_000)
    too_much = check(db, payments, "ord_1001", amount_minor=249_901)
    assert (too_much.outcome, too_much.reason) == ("blocked", "AMOUNT_EXCEEDS_REFUNDABLE")


@pytest.mark.parametrize(
    ("today", "outcome"),
    [(date(2026, 10, 31), "blocked"), (date(2026, 11, 1), "needs_review")],
    ids=["day_10", "day_11"],
)
def test_window_is_judged_on_the_day_given(
    db: Session, payments: ProviderClient, today: date, outcome: str
) -> None:
    # ord_1011 was delivered on 21 Oct (4 days before TODAY) and has no return yet.
    result = check_refund_eligibility(db, payments, owner("ord_1011"), "ord_1011", today=today)
    assert result.outcome == outcome


def test_refund_made_directly_at_the_provider_counts(db: Session, payments: ProviderClient) -> None:
    payments.http.post(
        "/refunds",
        json={
            "idempotency_key": "manual:1",
            "order_id": "ord_1001",
            "amount_minor": 249_900,
            "currency": "INR",
        },
    )
    assert check(db, payments, "ord_1001").reason == "ALREADY_REFUNDED"


def test_pending_refund_at_the_provider_counts(db: Session, payments: ProviderClient) -> None:
    payments.http.post("/admin/faults", json={"mode": "delay"})
    payments.http.post(
        "/refunds",
        json={
            "idempotency_key": "manual:2",
            "order_id": "ord_1001",
            "amount_minor": 100_000,
            "currency": "INR",
        },
    )
    result = check(db, payments, "ord_1001")
    assert (result.outcome, result.reason) == ("already_processing", "REFUND_PENDING")


@pytest.mark.parametrize("order_id", ["ord_1014", "ord_9999"], ids=["bobs", "missing"])
def test_someone_elses_or_missing_order(
    db: Session, payments: ProviderClient, order_id: str
) -> None:
    alice = AuthenticatedCustomer("cus_alice", "auth|alice")
    with pytest.raises(ServiceFailure) as error:
        check_refund_eligibility(db, payments, alice, order_id, today=TODAY)
    assert error.value.error_code == "ORDER_NOT_FOUND"


def test_no_decision_without_the_provider(db: Session) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down = ProviderClient(
        "payment", httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://x")
    )
    with pytest.raises(ServiceFailure) as error:
        check(db, down, "ord_1001")
    assert (error.value.error_code, error.value.retryable) == ("PROVIDER_UNAVAILABLE", True)


def test_provider_without_the_payment(db: Session, payments: ProviderClient) -> None:
    payments.http.post("/admin/reset", json={"payments": []})
    with pytest.raises(ServiceFailure) as error:
        check(db, payments, "ord_1001")
    assert error.value.error_code == "PAYMENT_NOT_FOUND"
