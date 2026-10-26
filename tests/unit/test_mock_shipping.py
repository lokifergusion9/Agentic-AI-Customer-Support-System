"""Mock shipping provider: courier scans, admin events and fault injection (W1.9)."""

import time
from collections.abc import Iterator
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_services.shipping.main import app, settings
from mock_services.shipping.store import STATUSES
from scripts.seed_data import push_mock_shipping
from tests.unit.live_server import serve

TODAY = date(2026, 10, 25)
KEY = {"X-API-Key": settings.mock_shipping_api_key}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app, headers=KEY) as client:
        push_mock_shipping(client, TODAY)
        yield client


def error_code(response: httpx.Response) -> str:
    body = response.json()
    assert body["success"] is False
    assert set(body) == {"success", "error_code", "retryable", "message"}
    return body["error_code"]


def add_event(client: TestClient, tracking_id: str, **body: object) -> httpx.Response:
    return client.post(f"/admin/shipments/{tracking_id}/events", json=body)


# --- API key and seed data --------------------------------------------------------------


def test_health_is_public() -> None:
    assert TestClient(app).get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/shipments/trk_1001"),
        ("GET", "/shipments?order_id=ord_1001"),
        ("POST", "/admin/reset"),
        ("POST", "/admin/faults"),
        ("POST", "/admin/shipments/trk_1001/events"),
    ],
)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}], ids=["missing", "wrong"])
def test_api_key_is_required(method: str, path: str, headers: dict[str, str]) -> None:
    response = TestClient(app).request(method, path, headers=headers, json={})
    assert response.status_code == 401
    assert error_code(response) == "INVALID_API_KEY"


def test_seed_push_loads_every_shipment(client: TestClient) -> None:
    assert push_mock_shipping(client, TODAY) == {"shipments": 15}


def test_delivered_shipment_has_the_full_scan_history(client: TestClient) -> None:
    shipment = client.get("/shipments/trk_1001").json()
    assert shipment["order_id"] == "ord_1001"
    assert [e["status"] for e in shipment["events"]] == [
        "label_created",
        "in_transit",
        "out_for_delivery",
        "delivered",
    ]
    assert shipment["status"] == "delivered"
    assert shipment["delivered_at"] == shipment["last_updated_at"] == shipment["events"][-1]["at"]


@pytest.mark.parametrize(
    ("tracking_id", "estimate", "last_update"),
    [
        ("trk_1008", "2026-10-22", "2026-10-21"),  # minor delay
        ("trk_1009", "2026-10-19", "2026-10-18"),  # significant delay
        ("trk_1010", "2026-10-14", "2026-10-13"),  # possibly lost: no scan for 12 days
    ],
)
def test_delayed_shipment_is_in_transit_with_its_last_scan(
    client: TestClient, tracking_id: str, estimate: str, last_update: str
) -> None:
    shipment = client.get(f"/shipments/{tracking_id}").json()
    assert (shipment["status"], shipment["delivered_at"]) == ("in_transit", None)
    assert shipment["estimated_delivery"] == estimate
    assert shipment["last_updated_at"].startswith(last_update)


def test_every_shipment_ends_with_its_current_status(client: TestClient) -> None:
    for number in range(1001, 1016):
        shipment = client.get(f"/shipments/trk_{number}").json()
        times = [e["at"] for e in shipment["events"]]
        assert shipment["events"][-1]["status"] == shipment["status"]
        assert times == sorted(times)
        assert {e["status"] for e in shipment["events"]} <= set(STATUSES)


# --- Queries ----------------------------------------------------------------------------


def test_shipments_by_order(client: TestClient) -> None:
    found = client.get("/shipments", params={"order_id": "ord_1009"}).json()
    assert [s["tracking_id"] for s in found] == ["trk_1009"]
    assert found[0] == client.get("/shipments/trk_1009").json()


def test_unknown_order_has_no_shipments(client: TestClient) -> None:
    assert client.get("/shipments", params={"order_id": "ord_9999"}).json() == []


def test_order_id_is_required(client: TestClient) -> None:
    assert client.get("/shipments").status_code == 422


def test_unknown_tracking_id_is_not_found(client: TestClient) -> None:
    response = client.get("/shipments/trk_9999")
    assert response.status_code == 404
    assert error_code(response) == "SHIPMENT_NOT_FOUND"


# --- Admin events -----------------------------------------------------------------------


def test_exception_event_with_a_new_estimate(client: TestClient) -> None:
    response = add_event(
        client,
        "trk_1009",
        status="exception",
        location="Pune hub",
        at="2026-10-24T12:00:00+05:30",
        estimated_delivery="2026-10-28",
    )
    assert response.status_code == 200
    shipment = client.get("/shipments/trk_1009").json()
    assert (shipment["status"], shipment["estimated_delivery"]) == ("exception", "2026-10-28")
    assert shipment["last_updated_at"] == "2026-10-24T12:00:00+05:30"
    assert shipment["events"][-1] == {
        "status": "exception",
        "at": "2026-10-24T12:00:00+05:30",
        "location": "Pune hub",
    }


def test_estimate_is_kept_when_the_courier_gives_none(client: TestClient) -> None:
    add_event(client, "trk_1008", status="out_for_delivery", location="Local delivery centre")
    assert client.get("/shipments/trk_1008").json()["estimated_delivery"] == "2026-10-22"


def test_event_time_defaults_to_now(client: TestClient) -> None:
    shipment = add_event(client, "trk_1008", status="exception", location="Hub").json()
    assert shipment["last_updated_at"] > "2026-10-25"


def test_delivered_event_sets_delivered_at(client: TestClient) -> None:
    shipment = add_event(client, "trk_1010", status="delivered", location="Customer").json()
    assert shipment["status"] == "delivered"
    assert shipment["delivered_at"] == shipment["last_updated_at"]


def test_delivered_shipment_takes_no_more_events(client: TestClient) -> None:
    response = add_event(client, "trk_1001", status="exception", location="Hub")
    assert response.status_code == 409
    assert error_code(response) == "SHIPMENT_DELIVERED"


def test_event_cannot_be_older_than_the_last_one(client: TestClient) -> None:
    response = add_event(
        client, "trk_1008", status="exception", location="Hub", at="2026-10-01T00:00:00+05:30"
    )
    assert response.status_code == 422
    assert error_code(response) == "EVENT_OUT_OF_ORDER"


def test_event_for_an_unknown_shipment(client: TestClient) -> None:
    response = add_event(client, "trk_9999", status="exception", location="Hub")
    assert error_code(response) == "SHIPMENT_NOT_FOUND"


@pytest.mark.parametrize(
    "body", [{"status": "lost", "location": "Hub"}, {"status": "exception", "location": ""}]
)
def test_invalid_event_is_rejected(client: TestClient, body: dict) -> None:
    assert add_event(client, "trk_1008", **body).status_code == 422


def test_reset_undoes_admin_events(client: TestClient) -> None:
    add_event(client, "trk_1008", status="exception", location="Hub")
    push_mock_shipping(client, TODAY)
    assert client.get("/shipments/trk_1008").json()["status"] == "in_transit"


# --- Fault injection --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [("GET", "/shipments/trk_1001"), ("GET", "/shipments?order_id=ord_1001")],
)
def test_error_fault_fails_the_next_reads(client: TestClient, method: str, path: str) -> None:
    client.post("/admin/faults", json={"mode": "error", "count": 2})
    for _ in range(2):
        response = client.request(method, path)
        assert response.status_code == 503
        assert error_code(response) == "PROVIDER_UNAVAILABLE"
        assert response.json()["retryable"] is True
    assert client.request(method, path).status_code == 200


def test_error_fault_with_a_custom_error(client: TestClient) -> None:
    client.post(
        "/admin/faults",
        json={
            "mode": "error",
            "status_code": 502,
            "error_code": "COURIER_DOWN",
            "retryable": False,
        },
    )
    response = client.get("/shipments/trk_1001")
    assert (response.status_code, error_code(response)) == (502, "COURIER_DOWN")
    assert response.json()["retryable"] is False


def test_delay_fault_slows_the_response(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "delay", "delay_seconds": 0.3})
    started = time.monotonic()
    response = client.get("/shipments/trk_1001")
    assert response.status_code == 200
    assert time.monotonic() - started >= 0.3


def test_admin_calls_are_not_faulted(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "error"})
    assert add_event(client, "trk_1008", status="exception", location="Hub").status_code == 200
    assert client.get("/shipments/trk_1008").status_code == 503


def test_faults_can_be_cleared_and_reset_clears_them(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "error", "count": 3})
    assert client.delete("/admin/faults").json() == {"queued": 0}
    assert client.get("/shipments/trk_1001").status_code == 200

    client.post("/admin/faults", json={"mode": "error"})
    push_mock_shipping(client, TODAY)
    assert client.get("/shipments/trk_1001").status_code == 200


@pytest.mark.parametrize(
    "body", [{"mode": "accept_then_timeout"}, {"mode": "error", "count": 0}, {"mode": "explode"}]
)
def test_invalid_faults_are_rejected(client: TestClient, body: dict) -> None:
    assert client.post("/admin/faults", json=body).status_code == 422


def test_client_times_out_then_the_next_read_works() -> None:
    with serve(app) as url, httpx.Client(base_url=url, headers=KEY, timeout=5) as admin:
        push_mock_shipping(admin, TODAY)
        admin.post("/admin/faults", json={"mode": "delay", "delay_seconds": 1})

        with httpx.Client(base_url=url, headers=KEY, timeout=0.3) as app_client:
            with pytest.raises(httpx.ReadTimeout):
                app_client.get("/shipments/trk_1009")
            assert app_client.get("/shipments/trk_1009").json()["status"] == "in_transit"
