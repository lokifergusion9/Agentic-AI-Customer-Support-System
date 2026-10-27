"""Run endpoints: start, read, timeline, clarification reply (W2.14)."""

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.runs import get_workflow_services
from app.auth.tokens import mint_token
from app.config import get_settings
from app.db.session import get_db
from app.main import app
from app.workflows.runner import WorkflowServices
from tests.integration import test_workflow as workflow
from tests.integration.test_workflow import HEADPHONES, Scenario, headphones_scenario, refund_ids

# The workflow tests' fixtures: seeded database, mock providers, scripted services.
db = workflow.db
providers = workflow.providers
make_services = workflow.make_services


def bearer(subject: str, role: str = "customer") -> dict[str, str]:
    return {"Authorization": f"Bearer {mint_token(subject, role)}"}


ALICE = bearer("auth|alice")
BOB = bearer("auth|bob")
REVIEWER = bearer("auth|reviewer_riya", "reviewer")


@pytest.fixture
def use_services(db: Session) -> Iterator[Callable[[WorkflowServices], TestClient]]:
    app.dependency_overrides[get_db] = lambda: db

    def install(services: WorkflowServices) -> TestClient:
        app.dependency_overrides[get_workflow_services] = lambda: services
        return TestClient(app)

    yield install
    app.dependency_overrides.clear()


@pytest.fixture
def client(use_services, make_services, providers, db: Session) -> TestClient:
    payments, _ = providers
    return use_services(make_services(headphones_scenario(*refund_ids(db, payments))))


def open_ticket(client: TestClient, message: str = HEADPHONES, order_id: str | None = "ord_1001"):
    body = {"message": message, **({"order_id": order_id} if order_id else {})}
    return client.post("/tickets", json=body, headers=ALICE).json()["id"]


def test_start_run_returns_the_paused_run_with_its_output(client: TestClient) -> None:
    ticket_id = open_ticket(client)

    response = client.post(f"/tickets/{ticket_id}/runs", headers=ALICE)

    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "waiting_for_approval"
    assert run["output"]["route"] == "request_approval"
    assert run["output"]["policy_references"][0] == "refund_policy_v1:section_3"


def test_another_customer_cannot_start_a_run_on_the_ticket(client: TestClient) -> None:
    ticket_id = open_ticket(client)
    response = client.post(f"/tickets/{ticket_id}/runs", headers=BOB)
    assert response.status_code == 404
    assert response.json()["error_code"] == "TICKET_NOT_FOUND"


def test_second_run_while_one_is_unfinished_conflicts(client: TestClient) -> None:
    ticket_id = open_ticket(client)
    client.post(f"/tickets/{ticket_id}/runs", headers=ALICE)

    response = client.post(f"/tickets/{ticket_id}/runs", headers=ALICE)

    assert response.status_code == 409
    assert response.json()["error_code"] == "RUN_ALREADY_ACTIVE"


def test_run_is_readable_by_its_customer_only(client: TestClient) -> None:
    run_id = client.post(f"/tickets/{open_ticket(client)}/runs", headers=ALICE).json()["id"]

    assert client.get(f"/runs/{run_id}", headers=ALICE).json()["id"] == run_id
    assert client.get(f"/runs/{run_id}", headers=BOB).status_code == 404


def test_timeline_is_redacted_and_visible_to_customer_and_reviewer(client: TestClient) -> None:
    run_id = client.post(f"/tickets/{open_ticket(client)}/runs", headers=ALICE).json()["id"]

    events = client.get(f"/runs/{run_id}/events", headers=ALICE).json()
    assert client.get(f"/runs/{run_id}/events", headers=REVIEWER).json() == events
    assert client.get(f"/runs/{run_id}/events", headers=BOB).status_code == 404

    types = [e["event_type"] for e in events]
    assert types[0] == "run_started" and "model_call" in types and "proposal_created" in types
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)
    assert "headphones" not in str(events).lower()


def test_clarification_reply_resumes_the_run(
    use_services, make_services, providers, db: Session
) -> None:
    payments, _ = providers
    base = headphones_scenario(*refund_ids(db, payments))
    plans = {
        **base.plans,
        "investigation": lambda call: (
            base.plans["investigation"]
            if "ord_1001" in call.messages[0].text
            else [[("list_orders", {})]]
        ),
    }
    outputs = {**base.outputs, "classifier": {**base.outputs["classifier"], "order_id": None}}
    client = use_services(make_services(Scenario(outputs, plans)))
    ticket_id = open_ticket(client, "Where is my refund?", None)
    run = client.post(f"/tickets/{ticket_id}/runs", headers=ALICE).json()
    assert run["status"] == "waiting_for_customer"

    bob = client.post(f"/runs/{run['id']}/messages", json={"message": "ord_1001"}, headers=BOB)
    assert bob.status_code == 404

    reply = client.post(f"/runs/{run['id']}/messages", json={"message": "ord_1001"}, headers=ALICE)
    assert reply.status_code == 200 and reply.json()["status"] == "waiting_for_approval"

    again = client.post(f"/runs/{run['id']}/messages", json={"message": "hi"}, headers=ALICE)
    assert again.status_code == 409 and again.json()["error_code"] == "RUN_NOT_WAITING"


def test_message_body_rejects_extra_fields(client: TestClient) -> None:
    response = client.post(
        "/runs/run_x/messages", json={"message": "hi", "customer_id": "cus_bob"}, headers=ALICE
    )
    assert response.status_code == 422


def test_unconfigured_model_is_a_clear_503(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    # Environment variables beat .env, so this holds even after a real key is configured.
    monkeypatch.setenv("LLM_MODEL", "")
    get_settings.cache_clear()
    app.dependency_overrides[get_db] = lambda: db
    try:
        client = TestClient(app)
        ticket_id = open_ticket(client)
        response = client.post(f"/tickets/{ticket_id}/runs", headers=ALICE)
    finally:
        app.dependency_overrides.clear()
        monkeypatch.undo()
        get_settings.cache_clear()
    assert response.status_code == 503
    assert "LLM_MODEL" in response.json()["detail"]
