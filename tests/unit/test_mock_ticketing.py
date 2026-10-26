"""Mock ticketing provider: retry-safe creates and updates, and fault injection (W1.10)."""

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_services.ticketing.main import app, settings
from scripts.seed_data import push_mock_ticketing
from tests.unit.live_server import serve

TODAY = date(2026, 10, 25)
KEY = {"X-API-Key": settings.mock_ticketing_api_key}
NEW = {"local_ticket_id": "tkt_new", "customer_id": "cus_alice", "subject": "Where is my refund?"}
RESOLVE = {"update_id": "tku:prp_1", "status": "resolved", "note": "Refund of Rs 2,499 issued."}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(app, headers=KEY) as client:
        push_mock_ticketing(client, TODAY)
        yield client


def error_code(response: httpx.Response) -> str:
    body = response.json()
    assert body["success"] is False
    assert set(body) == {"success", "error_code", "retryable", "message"}
    return body["error_code"]


def update(client: httpx.Client, ref: str = "ext_1003", **body: object) -> httpx.Response:
    return client.post(f"/tickets/{ref}/updates", json=RESOLVE | body)


def notes(client: httpx.Client, ref: str = "ext_1003") -> list[str]:
    return [n["text"] for n in client.get(f"/tickets/{ref}").json()["notes"]]


# --- API key and seed data --------------------------------------------------------------


def test_health_is_public() -> None:
    assert TestClient(app).get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/tickets/ext_1003"),
        ("POST", "/tickets"),
        ("POST", "/tickets/ext_1003/updates"),
        ("POST", "/admin/reset"),
        ("POST", "/admin/faults"),
    ],
)
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}], ids=["missing", "wrong"])
def test_api_key_is_required(method: str, path: str, headers: dict[str, str]) -> None:
    response = TestClient(app).request(method, path, headers=headers, json={})
    assert response.status_code == 401
    assert error_code(response) == "INVALID_API_KEY"


def test_seed_push_loads_the_earlier_refund_tickets(client: TestClient) -> None:
    assert push_mock_ticketing(client, TODAY) == {"tickets": 2}
    resolved = client.get("/tickets/ext_1002").json()
    assert (resolved["local_ticket_id"], resolved["status"]) == ("tkt_1002", "resolved")
    assert notes(client, "ext_1002") == ["Refund issued."]
    pending = client.get("/tickets/ext_1003").json()
    assert (pending["customer_id"], pending["status"]) == ("cus_bob", "open")
    assert pending["updated_at"] == pending["notes"][-1]["at"]


def test_unknown_ticket_is_not_found(client: TestClient) -> None:
    response = client.get("/tickets/ext_nope")
    assert response.status_code == 404
    assert error_code(response) == "TICKET_NOT_FOUND"


# --- Create -----------------------------------------------------------------------------


def test_create_ticket(client: TestClient) -> None:
    response = client.post("/tickets", json=NEW)
    assert response.status_code == 201
    ticket = response.json()
    assert ticket["ref"].startswith("ext_")
    assert (ticket["local_ticket_id"], ticket["status"], ticket["notes"]) == ("tkt_new", "open", [])
    assert client.get(f"/tickets/{ticket['ref']}").json() == ticket


def test_create_again_returns_the_same_ticket(client: TestClient) -> None:
    first = client.post("/tickets", json=NEW)
    second = client.post("/tickets", json=NEW)
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json() == second.json()


@pytest.mark.parametrize("change", [{"subject": "Other"}, {"customer_id": "cus_bob"}])
def test_create_with_changed_details_is_rejected(client: TestClient, change: dict) -> None:
    client.post("/tickets", json=NEW)
    response = client.post("/tickets", json=NEW | change)
    assert response.status_code == 409
    assert error_code(response) == "LOCAL_TICKET_ID_REUSED"


def test_seeded_ticket_cannot_be_created_twice(client: TestClient) -> None:
    body = {"local_ticket_id": "tkt_1003", "customer_id": "cus_bob"}
    replay = client.post("/tickets", json=body | {"subject": "Refund request for ord_1003"})
    assert (replay.status_code, replay.json()["ref"]) == (200, "ext_1003")


@pytest.mark.parametrize(
    "body",
    [NEW | {"subject": ""}, NEW | {"status": "closed"}, {"local_ticket_id": "tkt_x"}],
    ids=["empty_subject", "bad_status", "missing_fields"],
)
def test_invalid_create_is_rejected(client: TestClient, body: dict) -> None:
    assert client.post("/tickets", json=body).status_code == 422


# --- Updates ----------------------------------------------------------------------------


def test_update_changes_status_and_adds_a_note(client: TestClient) -> None:
    response = update(client)
    assert response.status_code == 201
    assert response.json()["update_id"] == "tku:prp_1"
    ticket = client.get("/tickets/ext_1003").json()
    assert ticket["status"] == "resolved"
    assert notes(client)[-1] == RESOLVE["note"]
    assert ticket["updated_at"] == response.json()["applied_at"]


def test_status_only_and_note_only_updates(client: TestClient) -> None:
    update(client, update_id="u1", status="escalated", note=None)
    assert client.get("/tickets/ext_1003").json()["status"] == "escalated"
    assert len(notes(client)) == 1
    update(client, update_id="u2", status=None, note="Customer called.")
    assert client.get("/tickets/ext_1003").json()["status"] == "escalated"
    assert notes(client)[-1] == "Customer called."


def test_same_update_again_is_applied_once(client: TestClient) -> None:
    first = update(client)
    second = update(client)
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json() == second.json()
    assert notes(client).count(RESOLVE["note"]) == 1


@pytest.mark.parametrize(
    ("ref", "change"),
    [("ext_1003", {"note": "Other"}), ("ext_1003", {"status": "escalated"}), ("ext_1002", {})],
    ids=["note", "status", "other_ticket"],
)
def test_update_id_with_a_changed_update_is_rejected(
    client: TestClient, ref: str, change: dict
) -> None:
    update(client)
    response = update(client, ref, **change)
    assert response.status_code == 409
    assert error_code(response) == "UPDATE_ID_REUSED"


def test_update_for_an_unknown_ticket(client: TestClient) -> None:
    assert error_code(update(client, "ext_nope")) == "TICKET_NOT_FOUND"


@pytest.mark.parametrize(
    "body",
    [
        {"status": None, "note": None},
        {"status": "closed"},
        {"note": ""},
        {"update_id": ""},
    ],
    ids=["no_change", "bad_status", "empty_note", "empty_update_id"],
)
def test_invalid_update_is_rejected(client: TestClient, body: dict) -> None:
    assert update(client, **body).status_code == 422


def test_concurrent_identical_updates_apply_once(client: TestClient) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: update(client), range(8)))
    assert sorted(r.status_code for r in responses) == [200] * 7 + [201]
    assert notes(client).count(RESOLVE["note"]) == 1


def test_reset_forgets_updates(client: TestClient) -> None:
    update(client)
    push_mock_ticketing(client, TODAY)
    assert client.get("/tickets/ext_1003").json()["status"] == "open"
    assert update(client).status_code == 201


# --- Fault injection --------------------------------------------------------------------


def test_refund_succeeds_ticket_update_fails_then_only_the_update_is_retried(
    client: TestClient,
) -> None:
    """PDF section 14: retry only the ticket update, with the same update_id."""
    client.post("/admin/faults", json={"mode": "error"})
    failed = update(client)
    assert failed.status_code == 503
    assert failed.json()["retryable"] is True
    assert client.get("/tickets/ext_1003").json()["status"] == "open"

    retried = update(client)
    assert retried.status_code == 201
    assert client.get("/tickets/ext_1003").json()["status"] == "resolved"
    assert notes(client).count(RESOLVE["note"]) == 1


def test_error_fault_on_create_records_nothing(client: TestClient) -> None:
    client.post(
        "/admin/faults",
        json={"mode": "error", "status_code": 500, "error_code": "HELPDESK_DOWN"},
    )
    assert error_code(client.post("/tickets", json=NEW)) == "HELPDESK_DOWN"
    assert client.post("/tickets", json=NEW).status_code == 201


def test_reads_and_replays_are_not_faulted(client: TestClient) -> None:
    update(client)
    client.post("/admin/faults", json={"mode": "error"})
    assert client.get("/tickets/ext_1003").status_code == 200
    assert update(client).status_code == 200
    # The fault is still queued for the next new write.
    assert update(client, update_id="u2").status_code == 503


def test_delay_fault_applies_after_the_wait(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "delay", "delay_seconds": 0.3})
    started = time.monotonic()
    assert update(client).status_code == 201
    assert time.monotonic() - started >= 0.3
    assert client.get("/tickets/ext_1003").json()["status"] == "resolved"


def test_retry_during_a_delay_is_applied_once(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "delay", "delay_seconds": 0.5})
    with ThreadPoolExecutor(max_workers=1) as pool:
        slow = pool.submit(update, client)
        time.sleep(0.1)
        retry = update(client)  # not faulted: the slow request took the only fault
        assert retry.status_code == 201
        assert slow.result().status_code == 200
    assert notes(client).count(RESOLVE["note"]) == 1


def test_faults_can_be_cleared_and_reset_clears_them(client: TestClient) -> None:
    client.post("/admin/faults", json={"mode": "error", "count": 3})
    assert client.delete("/admin/faults").json() == {"queued": 0}
    assert update(client).status_code == 201

    client.post("/admin/faults", json={"mode": "error"})
    push_mock_ticketing(client, TODAY)
    assert update(client).status_code == 201


@pytest.mark.parametrize("body", [{"mode": "explode"}, {"mode": "delay", "delay_seconds": -1}])
def test_invalid_faults_are_rejected(client: TestClient, body: dict) -> None:
    assert client.post("/admin/faults", json=body).status_code == 422


def test_update_applied_but_client_timed_out_is_safe_to_retry() -> None:
    with serve(app) as url, httpx.Client(base_url=url, headers=KEY, timeout=5) as admin:
        push_mock_ticketing(admin, TODAY)
        admin.post("/admin/faults", json={"mode": "accept_then_timeout", "delay_seconds": 1})

        with httpx.Client(base_url=url, headers=KEY, timeout=0.3) as app_client:
            with pytest.raises(httpx.ReadTimeout):
                update(app_client)
            # The update was applied; retrying it returns the stored result.
            retry = update(app_client)
            assert retry.status_code == 200
            assert app_client.get("/tickets/ext_1003").json()["status"] == "resolved"
            assert notes(app_client).count(RESOLVE["note"]) == 1
