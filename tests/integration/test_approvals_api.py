"""Approval endpoints: the reviewer's queue and deciding one approval (W3.4)."""

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from tests.integration import test_runs_api as api
from tests.integration import test_workflow as workflow
from tests.integration.test_runs_api import ALICE, REVIEWER, bearer, open_ticket

# The workflow tests' fixtures: seeded database, mock providers, scripted services.
db = workflow.db
providers = workflow.providers
make_services = workflow.make_services
use_services = api.use_services
client = api.client

SENIOR = bearer("auth|senior_sam", "senior_reviewer")


def paused_refund(client: TestClient) -> dict:
    run = client.post(f"/tickets/{open_ticket(client)}/runs", headers=ALICE).json()
    assert run["status"] == "waiting_for_approval"
    return run


def decide(client: TestClient, approval_id: str, decision: str, headers=REVIEWER):
    return client.post(
        f"/approvals/{approval_id}/decision", json={"decision": decision}, headers=headers
    )


def test_queue_lists_the_paused_refund_for_the_reviewer(client: TestClient) -> None:
    run = paused_refund(client)

    response = client.get("/approvals", headers=REVIEWER)

    assert response.status_code == 200
    [pending] = response.json()
    assert pending["approval_id"] == run["output"]["approval_id"]
    assert (pending["run_id"], pending["order_id"]) == (run["id"], "ord_1001")
    assert (pending["amount_minor"], pending["currency"]) == (249900, "INR")
    assert pending["policy_refs"][0] == "refund_policy_v1:section_3"
    assert pending["complaint_category"] == "refund_request"
    assert pending["customer_name"] and pending["expires_at"]


def test_customers_and_anonymous_callers_cannot_see_the_queue(client: TestClient) -> None:
    assert client.get("/approvals", headers=ALICE).status_code == 403
    assert client.get("/approvals").status_code == 401


def test_approving_resumes_the_run_and_pays_once(
    client: TestClient, providers, db: Session
) -> None:
    payments, _ = providers
    run = paused_refund(client)
    approval_id = run["output"]["approval_id"]

    response = decide(client, approval_id, "approved")

    assert response.status_code == 200
    body = response.json()
    assert body["approval"]["decision"] == "approved"
    assert body["approval"]["reviewer_id"] == "auth|reviewer_riya"  # from the token
    assert body["run"]["id"] == run["id"]
    assert (body["run"]["status"], body["run"]["output"]["route"]) == ("completed", "refunded")
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1
    assert client.get("/approvals", headers=REVIEWER).json() == []

    # A second decision is refused and pays nothing more.
    again = decide(client, approval_id, "approved", headers=SENIOR)
    assert (again.status_code, again.json()["error_code"]) == (409, "APPROVAL_ALREADY_DECIDED")
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1


def test_rejecting_ends_the_run_without_a_refund(client: TestClient, providers) -> None:
    payments, _ = providers
    run = paused_refund(client)

    body = decide(client, run["output"]["approval_id"], "rejected").json()

    assert (body["run"]["status"], body["run"]["output"]["route"]) == ("completed", "rejected")
    assert payments.get("/payments/ord_1001")["refunds"] == []
    run_view = client.get(f"/runs/{run['id']}", headers=ALICE).json()
    assert run_view["status"] == "completed"


def test_reviewer_identity_cannot_come_from_the_body(client: TestClient) -> None:
    run = paused_refund(client)

    response = client.post(
        f"/approvals/{run['output']['approval_id']}/decision",
        json={"decision": "approved", "reviewer_id": "auth|senior_sam"},
        headers=REVIEWER,
    )

    assert response.status_code == 422


def test_customer_cannot_approve_their_own_refund(client: TestClient, providers) -> None:
    payments, _ = providers
    run = paused_refund(client)

    response = decide(client, run["output"]["approval_id"], "approved", headers=ALICE)

    assert response.status_code == 403
    assert payments.get("/payments/ord_1001")["refunds"] == []


def test_unknown_approval_is_404(client: TestClient) -> None:
    response = decide(client, "apr_nope", "approved")
    assert (response.status_code, response.json()["error_code"]) == (404, "APPROVAL_NOT_FOUND")


def test_expired_approval_is_refused_and_the_run_escalates(
    client: TestClient, providers, db: Session
) -> None:
    payments, _ = providers
    run = paused_refund(client)
    db.execute(
        text("UPDATE support.approvals SET expires_at = created_at WHERE id = :a"),
        {"a": run["output"]["approval_id"]},
    )
    db.commit()

    response = decide(client, run["output"]["approval_id"], "approved")

    assert (response.status_code, response.json()["error_code"]) == (409, "APPROVAL_EXPIRED")
    assert payments.get("/payments/ord_1001")["refunds"] == []
    assert client.get(f"/runs/{run['id']}", headers=ALICE).json()["status"] == "escalated"
