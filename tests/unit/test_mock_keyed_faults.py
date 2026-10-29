"""Faults keyed to a logical operation, an order and an occurrence (W4.4, E§22)."""

from collections.abc import Iterator
from datetime import date

import pytest
from fastapi.testclient import TestClient

from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from mock_services.ticketing.main import app as ticketing_app
from mock_services.ticketing.main import settings as ticketing_settings
from scripts.seed_data import push_mock_payments, push_mock_shipping, push_mock_ticketing

TODAY = date(2026, 10, 25)


@pytest.fixture
def payments() -> Iterator[TestClient]:
    key = {"X-API-Key": payments_settings.mock_payments_api_key}
    with TestClient(payments_app, headers=key) as client:
        push_mock_payments(client, TODAY)
        yield client


@pytest.fixture
def shipping() -> Iterator[TestClient]:
    key = {"X-API-Key": shipping_settings.mock_shipping_api_key}
    with TestClient(shipping_app, headers=key) as client:
        push_mock_shipping(client, TODAY)
        yield client


@pytest.fixture
def ticketing() -> Iterator[TestClient]:
    key = {"X-API-Key": ticketing_settings.mock_ticketing_api_key}
    with TestClient(ticketing_app, headers=key) as client:
        push_mock_ticketing(client, TODAY)
        yield client


def refund(order_id: str, amount: int, key: str) -> dict:
    return {"idempotency_key": key, "order_id": order_id, "amount_minor": amount, "currency": "INR"}


def test_refund_fault_fires_on_its_order_and_occurrence_only(payments: TestClient) -> None:
    fault = {"operation": "refund.create", "match": {"order_id": "ord_1001"}, "occurrence": 2}
    assert payments.post("/admin/faults", json={**fault, "mode": "error"}).status_code == 200

    # Another order's refund neither fails nor counts as an occurrence.
    assert payments.post("/refunds", json=refund("ord_1004", 129_900, "k0")).status_code == 201
    # ord_1001: the first call goes through, the second fails, the third goes through.
    assert payments.post("/refunds", json=refund("ord_1001", 100_000, "k1")).status_code == 201
    assert payments.post("/refunds", json=refund("ord_1001", 100_000, "k2")).status_code == 503
    assert payments.post("/refunds", json=refund("ord_1001", 49_900, "k3")).status_code == 201


def test_a_keyed_fault_is_independent_of_the_path_taken(payments: TestClient) -> None:
    """Two agents call the provider in a different order and still meet the same failure."""
    fault = {"operation": "refund.create", "match": {"order_id": "ord_1001"}, "mode": "error"}
    outcomes = []
    for path in (["ord_1004", "ord_1001"], ["ord_1001", "ord_1004"]):
        push_mock_payments(payments, TODAY)
        payments.post("/admin/faults", json=fault)
        amounts = {"ord_1001": 249_900, "ord_1004": 129_900}
        codes = {
            oid: payments.post("/refunds", json=refund(oid, amounts[oid], f"k-{oid}")).status_code
            for oid in path
        }
        outcomes.append(codes)
    assert outcomes[0] == outcomes[1] == {"ord_1001": 503, "ord_1004": 201}


def test_payment_reads_can_fail_for_several_calls(payments: TestClient) -> None:
    fault = {"operation": "payment.get", "match": {"order_id": "ord_1001"}, "count": 2}
    payments.post("/admin/faults", json={**fault, "mode": "error"})
    assert payments.get("/payments/ord_1004").status_code == 200
    assert [payments.get("/payments/ord_1001").status_code for _ in range(3)] == [503, 503, 200]


def test_reads_never_consume_the_plain_queue(payments: TestClient) -> None:
    # The plain queue keeps its old meaning: the next refund request, not the next read.
    payments.post("/admin/faults", json={"mode": "error"})
    assert payments.get("/payments/ord_1001").status_code == 200
    assert payments.post("/refunds", json=refund("ord_1001", 249_900, "k")).status_code == 503


def test_shipment_reads_match_by_order_either_way(shipping: TestClient) -> None:
    fault = {"operation": "shipment.read", "match": {"order_id": "ord_1008"}, "count": 2}
    shipping.post("/admin/faults", json={**fault, "mode": "error"})
    assert shipping.get("/shipments", params={"order_id": "ord_1011"}).status_code == 200
    assert shipping.get("/shipments/trk_1008").status_code == 503  # by tracking ID
    assert shipping.get("/shipments", params={"order_id": "ord_1008"}).status_code == 503
    assert shipping.get("/shipments/trk_1008").status_code == 200


def test_ticket_updates_fail_by_operation(ticketing: TestClient) -> None:
    ticketing.post("/admin/faults", json={"operation": "ticket.update", "mode": "error"})
    update = {"update_id": "tku:t1", "note": "Refund confirmed."}
    assert ticketing.post("/tickets/ext_1002/updates", json=update).status_code == 503
    assert ticketing.post("/tickets/ext_1002/updates", json=update).status_code in (200, 201)


def test_unknown_operations_and_stray_match_are_rejected(payments: TestClient) -> None:
    unknown = payments.post("/admin/faults", json={"operation": "refund.delete", "mode": "error"})
    assert (unknown.status_code, unknown.json()["error_code"]) == (422, "UNKNOWN_OPERATION")
    stray = payments.post("/admin/faults", json={"mode": "error", "match": {"order_id": "x"}})
    assert (stray.status_code, stray.json()["error_code"]) == (422, "INVALID_FAULT")


def test_reset_clears_keyed_faults_and_their_counts(payments: TestClient) -> None:
    fault = {"operation": "payment.get", "match": {"order_id": "ord_1001"}, "occurrence": 2}
    payments.post("/admin/faults", json={**fault, "mode": "error"})
    payments.get("/payments/ord_1001")  # counts as the first occurrence
    push_mock_payments(payments, TODAY)
    assert payments.get("/payments/ord_1001").status_code == 200
    assert payments.get("/payments/ord_1001").status_code == 200
