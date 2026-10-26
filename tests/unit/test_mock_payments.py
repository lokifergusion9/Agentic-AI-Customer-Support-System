"""Mock payment provider: idempotency contract, queries and fault injection (W1.8)."""

import socket
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from mock_services.payments.main import app, settings
from scripts.seed_data import push_mock_payments

TODAY = date(2026, 10, 25)
KEY = {"X-API-Key": settings.mock_payments_api_key}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app, headers=KEY) as client:
        push_mock_payments(client, TODAY)
        yield client


def refund(key: str = "refund:prp_t1", order_id: str = "ord_1001", amount: int = 249_900):
    return {"idempotency_key": key, "order_id": order_id, "amount_minor": amount, "currency": "INR"}


def error_code(response: httpx.Response) -> str:
    body = response.json()
    assert body["success"] is False and set(body) == {
        "success",
        "error_code",
        "retryable",
        "message",
    }
    return body["error_code"]


# --- API key and seed data --------------------------------------------------------------


def test_health_is_public() -> None:
    assert TestClient(app).get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/payments/ord_1001"),
        ("GET", "/refunds/rf_1002"),
        ("POST", "/refunds"),
        ("POST", "/admin/reset"),
        ("POST", "/admin/faults"),
    ],
)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}], ids=["missing", "wrong"])
def test_api_key_is_required(method: str, path: str, headers: dict[str, str]) -> None:
    response = TestClient(app).request(method, path, headers=headers, json={})
    assert response.status_code == 401
    assert error_code(response) == "INVALID_API_KEY"


def test_seed_push_loads_every_payment(client: TestClient) -> None:
    assert push_mock_payments(client, TODAY) == {"payments": 15, "refunds": 2}
    payment = client.get("/payments/ord_1001").json()
    assert (payment["status"], payment["amount_minor"], payment["refundable_minor"]) == (
        "captured",
        249_900,
        249_900,
    )


# --- Queries ----------------------------------------------------------------------------


def test_past_refunds_reduce_the_refundable_amount(client: TestClient) -> None:
    for order_id, status in [("ord_1002", "succeeded"), ("ord_1003", "pending")]:
        payment = client.get(f"/payments/{order_id}").json()
        assert payment["refundable_minor"] == 0
        assert [r["status"] for r in payment["refunds"]] == [status]


def test_refund_can_be_found_by_order_key_and_reference(client: TestClient) -> None:
    by_order = client.get("/refunds", params={"order_id": "ord_1002"}).json()
    by_key = client.get("/refunds", params={"idempotency_key": "refund:prp_1002"}).json()
    by_reference = client.get("/refunds/rf_1002").json()
    assert by_order == by_key == [by_reference]
    assert by_reference["provider_reference"] == "rf_1002"


def test_unknown_key_returns_an_empty_list(client: TestClient) -> None:
    assert client.get("/refunds", params={"idempotency_key": "refund:unknown"}).json() == []


@pytest.mark.parametrize("params", [{}, {"order_id": "ord_1001", "idempotency_key": "k"}])
def test_refund_query_needs_exactly_one_filter(client: TestClient, params: dict) -> None:
    response = client.get("/refunds", params=params)
    assert response.status_code == 422
    assert error_code(response) == "INVALID_QUERY"


@pytest.mark.parametrize(
    ("path", "code"),
    [("/payments/ord_9999", "PAYMENT_NOT_FOUND"), ("/refunds/rf_nope", "REFUND_NOT_FOUND")],
)
def test_unknown_ids_are_not_found(client: TestClient, path: str, code: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert error_code(response) == code


# --- Idempotency contract ---------------------------------------------------------------


def test_new_refund_succeeds(client: TestClient) -> None:
    response = client.post("/refunds", json=refund())
    assert response.status_code == 201
    body = response.json()
    assert body["provider_reference"].startswith("rf_")
    assert (body["status"], body["amount_minor"]) == ("succeeded", 249_900)
    assert client.get("/payments/ord_1001").json()["refundable_minor"] == 0


def test_same_key_and_payload_returns_the_same_refund(client: TestClient) -> None:
    first = client.post("/refunds", json=refund())
    second = client.post("/refunds", json=refund())
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json() == second.json()
    assert len(client.get("/refunds", params={"order_id": "ord_1001"}).json()) == 1


@pytest.mark.parametrize(
    "change",
    [{"amount_minor": 100}, {"order_id": "ord_1014"}, {"currency": "USD"}],
    ids=["amount", "order", "currency"],
)
def test_same_key_with_a_changed_payload_is_rejected(client: TestClient, change: dict) -> None:
    client.post("/refunds", json=refund())
    response = client.post("/refunds", json=refund() | change)
    assert response.status_code == 409
    assert error_code(response) == "IDEMPOTENCY_KEY_REUSED"
    assert len(client.get("/refunds", params={"order_id": "ord_1001"}).json()) == 1


def test_concurrent_requests_with_one_key_create_one_refund(client: TestClient) -> None:
    body = refund(order_id="ord_1014", amount=8_999_900)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: client.post("/refunds", json=body), range(8)))
    assert sorted(r.status_code for r in responses) == [200] * 7 + [201]
    assert len({r.json()["provider_reference"] for r in responses}) == 1
    assert len(client.get("/refunds", params={"order_id": "ord_1014"}).json()) == 1


def test_partial_refunds_up_to_the_paid_amount(client: TestClient) -> None:
    assert client.post("/refunds", json=refund("k1", amount=200_000)).status_code == 201
    over = client.post("/refunds", json=refund("k2", amount=49_901))
    assert error_code(over) == "AMOUNT_EXCEEDS_REFUNDABLE"
    assert client.post("/refunds", json=refund("k3", amount=49_900)).status_code == 201


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        (refund(order_id="ord_9999"), 404, "PAYMENT_NOT_FOUND"),
        (refund() | {"currency": "USD"}, 422, "CURRENCY_MISMATCH"),
        (refund(order_id="ord_1002", amount=1), 422, "AMOUNT_EXCEEDS_REFUNDABLE"),
        (refund(order_id="ord_1003", amount=1), 422, "AMOUNT_EXCEEDS_REFUNDABLE"),
    ],
    ids=["no_payment", "currency", "already_refunded", "refund_pending"],
)
def test_invalid_refunds_are_rejected(
    client: TestClient, body: dict, status: int, code: str
) -> None:
    response = client.post("/refunds", json=body)
    assert response.status_code == status
    assert error_code(response) == code


@pytest.mark.parametrize("amount", [0, -1])
def test_amount_must_be_positive(client: TestClient, amount: int) -> None:
    assert client.post("/refunds", json=refund(amount=amount)).status_code == 422


# --- Fault injection --------------------------------------------------------------------


def test_error_fault_records_nothing(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "error", "count": 2})
    for _ in range(2):
        response = client.post("/refunds", json=refund())
        assert response.status_code == 503
        assert error_code(response) == "PROVIDER_UNAVAILABLE"
        assert response.json()["retryable"] is True
    assert client.get("/refunds", params={"order_id": "ord_1001"}).json() == []
    # The faults are used up, so a retry with the same key now succeeds.
    assert client.post("/refunds", json=refund()).status_code == 201


def test_error_fault_with_a_custom_error(client: TestClient) -> None:
    client.post(
        "/admin/faults",
        json={"mode": "error", "status_code": 400, "error_code": "CARD_CLOSED", "retryable": False},
    )
    response = client.post("/refunds", json=refund())
    assert (response.status_code, error_code(response)) == (400, "CARD_CLOSED")
    assert response.json()["retryable"] is False


def test_delay_fault_leaves_the_refund_pending_until_settled(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "delay"})
    created = client.post("/refunds", json=refund())
    assert (created.status_code, created.json()["status"]) == (201, "pending")
    assert client.get("/payments/ord_1001").json()["refundable_minor"] == 0

    reference = created.json()["provider_reference"]
    settled = client.post(f"/admin/refunds/{reference}/settle", json={"status": "succeeded"})
    assert settled.json()["status"] == "succeeded"
    assert client.get(f"/refunds/{reference}").json()["status"] == "succeeded"


def test_failed_settlement_frees_the_amount(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "delay"})
    reference = client.post("/refunds", json=refund()).json()["provider_reference"]
    client.post(f"/admin/refunds/{reference}/settle", json={"status": "failed"})
    assert client.get("/payments/ord_1001").json()["refundable_minor"] == 249_900


def test_only_pending_refunds_can_be_settled(client: TestClient) -> None:
    response = client.post("/admin/refunds/rf_1002/settle", json={"status": "failed"})
    assert response.status_code == 409
    assert error_code(response) == "REFUND_NOT_PENDING"


def test_replay_is_never_faulted(client: TestClient) -> None:
    first = client.post("/refunds", json=refund())
    client.post("/admin/faults", json={"mode": "error"})
    assert client.post("/refunds", json=refund()).json() == first.json()
    # The fault is still queued for the next new refund.
    assert client.post("/refunds", json=refund("k2", "ord_1014", 1)).status_code == 503


def test_faults_can_be_cleared_and_reset_clears_them(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "error", "count": 3})
    assert client.delete("/admin/faults").json() == {"queued": 0}
    assert client.post("/refunds", json=refund()).status_code == 201

    client.post("/admin/faults", json={"mode": "error"})
    push_mock_payments(client, TODAY)
    assert client.post("/refunds", json=refund()).status_code == 201


@pytest.mark.parametrize(
    "body",
    [{"mode": "explode"}, {"mode": "error", "count": 0}, {"mode": "error", "status_code": 200}],
)
def test_invalid_faults_are_rejected(client: TestClient, body: dict) -> None:
    assert client.post("/admin/faults", json=body).status_code == 422


# --- Accept, then time out (PDF section 12) ---------------------------------------------


@pytest.fixture
def live_server() -> Iterator[str]:
    """The mock on a real socket, so a real client timeout can fire."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "mock payment server did not start"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_refund_is_recorded_even_when_the_client_times_out(live_server: str) -> None:
    with httpx.Client(base_url=live_server, headers=KEY, timeout=5) as admin:
        push_mock_payments(admin, TODAY)
        admin.post("/admin/faults", json={"mode": "accept_then_timeout", "delay_seconds": 1})

    with httpx.Client(base_url=live_server, headers=KEY, timeout=0.3) as app_client:
        with pytest.raises(httpx.ReadTimeout):
            app_client.post("/refunds", json=refund())

        # Reconcile by idempotency key while the provider is still holding the response.
        found = app_client.get("/refunds", params={"idempotency_key": "refund:prp_t1"}).json()
        assert [r["status"] for r in found] == ["succeeded"]

        # Retrying with the same key and payload returns that refund instead of a second one.
        retry = app_client.post("/refunds", json=refund())
        assert retry.status_code == 200
        assert retry.json()["provider_reference"] == found[0]["provider_reference"]
        assert len(app_client.get("/refunds", params={"order_id": "ord_1001"}).json()) == 1
