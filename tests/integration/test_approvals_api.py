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


def test_expired_approval_is_refused_and_a_new_one_is_requested(
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
    after = client.get(f"/runs/{run['id']}", headers=ALICE).json()
    assert after["status"] == "waiting_for_approval"
    [renewed] = client.get("/approvals", headers=REVIEWER).json()
    assert renewed["approval_id"] == after["output"]["approval_id"] != run["output"]["approval_id"]
    assert renewed["proposal_id"] == run["output"]["proposal_id"]


# --- W3.5: the approval lifecycle, end to end and in order ---------------------------------


def test_approval_lifecycle_runs_in_order(client: TestClient, providers) -> None:
    payments, _ = providers
    run = paused_refund(client)

    decide(client, run["output"]["approval_id"], "approved")

    events = client.get(f"/runs/{run['id']}/events", headers=REVIEWER).json()
    types = [e["event_type"] for e in events]
    statuses = [e["payload"]["status"] for e in events if e["event_type"] == "run_status_changed"]
    lifecycle = [
        "proposal_created",  # 1-2: immutable proposal and its pending approval
        "approval_decided",  # 4-6: authenticated, permitted reviewer; decision saved
        "run_resumed",  # 7: the same workflow
        "eligibility_rechecked",  # 8: eligibility and refundable balance
        "refund_reserved",  # 9: only the approved action
        "refund_executed",
    ]
    assert [types.index(name) for name in lifecycle] == sorted(
        types.index(name) for name in lifecycle
    )
    # 3: paused before the decision, running again only after it.
    assert statuses == ["waiting_for_approval", "running", "completed"]
    decided = events[types.index("approval_decided")]
    assert (decided["actor"], decided["payload"]["reviewer_role"]) == (
        "reviewer:auth|reviewer_riya",
        "reviewer",
    )
    approved_amount = next(
        e["payload"]["amount_minor"] for e in events if e["event_type"] == "proposal_created"
    )
    paid = payments.get("/payments/ord_1001")["refunds"]
    assert [r["amount_minor"] for r in paid] == [approved_amount]


def refund_elsewhere(payments, amount_minor: int) -> None:
    """A refund made outside this run while the approval waited."""
    payments.request(
        "POST",
        "/refunds",
        json={
            "idempotency_key": f"elsewhere:{amount_minor}",
            "order_id": "ord_1001",
            "amount_minor": amount_minor,
            "currency": "INR",
        },
    )


def proposal_status(db: Session, proposal_id: str) -> str:
    return db.execute(
        text("SELECT status FROM support.action_proposals WHERE id = :id"), {"id": proposal_id}
    ).scalar_one()


# --- W3.6: invalidation ----------------------------------------------------------------------


def test_nothing_left_to_refund_invalidates_the_approval(
    client: TestClient, providers, db: Session
) -> None:
    payments, _ = providers
    run = paused_refund(client)
    refund_elsewhere(payments, 249900)

    body = decide(client, run["output"]["approval_id"], "approved").json()

    assert body["run"]["status"] == "escalated"
    assert body["run"]["output"]["validation_problems"] == ["NOT_ELIGIBLE"]
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1  # only the outside one
    assert proposal_status(db, run["output"]["proposal_id"]) == "invalidated"
    types = [
        e["event_type"] for e in client.get(f"/runs/{run['id']}/events", headers=REVIEWER).json()
    ]
    assert {"proposal_details_changed", "proposal_invalidated"} <= set(types)
    assert "refund_reserved" not in types


def test_changed_amount_needs_a_new_approval(client: TestClient, providers, db: Session) -> None:
    payments, _ = providers
    run = paused_refund(client)
    refund_elsewhere(payments, 100000)  # ₹1,000 refunded elsewhere: ₹1,499 is left

    body = decide(client, run["output"]["approval_id"], "approved").json()

    # The approval of ₹2,499 authorizes nothing; ₹1,499 is proposed for a new approval.
    assert body["run"]["status"] == "waiting_for_approval"
    assert proposal_status(db, run["output"]["proposal_id"]) == "invalidated"
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1
    [pending] = client.get("/approvals", headers=REVIEWER).json()
    assert pending["proposal_id"] != run["output"]["proposal_id"]
    assert pending["amount_minor"] == 149900
    assert "₹1,499.00" in body["run"]["output"]["final_response"]

    done = decide(client, pending["approval_id"], "approved").json()
    assert (done["run"]["status"], done["run"]["output"]["route"]) == ("completed", "refunded")
    paid = [r["amount_minor"] for r in payments.get("/payments/ord_1001")["refunds"]]
    assert sorted(paid) == [100000, 149900]
