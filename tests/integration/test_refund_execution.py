"""Executing an approved refund: verified, reserved, sent once, outcome persisted (W1.15)."""

from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.services.approvals import decide_approval
from app.services.eligibility import check_refund_eligibility
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient
from app.services.refunds import execute_approved_refund, idempotency_key
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from scripts.seed_data import push_mock_payments, seed
from tests.integration.test_approvals import RIYA, SAM
from tests.integration.test_proposals import ALICE, BOB, NOW, TODAY, propose
from tests.unit.live_server import serve

HEADERS = {"X-API-Key": payments_settings.mock_payments_api_key}


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    seed(conn, TODAY)
    with Session(bind=conn) as session:
        yield session


@pytest.fixture
def payments() -> Iterator[ProviderClient]:
    with TestClient(payments_app, headers=HEADERS) as http:
        push_mock_payments(http, TODAY)
        yield ProviderClient("payment", http)


def approved(db: Session, payments: ProviderClient, customer=ALICE, order_id="ord_1001"):
    _, _, proposal = propose(db, payments, customer, order_id)
    reviewer = SAM if proposal.required_role == "senior_reviewer" else RIYA
    decide_approval(db, reviewer, proposal.approval_id, "approved", now=NOW)
    return proposal


def provider_refunds(payments: ProviderClient, order_id: str) -> list[dict]:
    return payments.get("/refunds", params={"order_id": order_id})


def proposal_status(db: Session, proposal_id: str) -> str:
    return db.execute(
        text("SELECT status FROM support.action_proposals WHERE id = :id"), {"id": proposal_id}
    ).scalar_one()


def operations(db: Session, order_id: str) -> list[str]:
    return list(
        db.execute(
            text("SELECT status FROM support.refund_operations WHERE order_id = :order"),
            {"order": order_id},
        ).scalars()
    )


def failure(call) -> ServiceFailure:
    with pytest.raises(ServiceFailure) as error:
        call()
    return error.value


# --- The normal path --------------------------------------------------------------------


def test_approved_refund_is_executed(db: Session, payments: ProviderClient) -> None:
    proposal = approved(db, payments)

    operation = execute_approved_refund(db, payments, proposal.proposal_id)

    assert (operation.status, operation.amount_minor, operation.order_id) == (
        "succeeded",
        249_900,
        "ord_1001",
    )
    assert operation.idempotency_key == idempotency_key(proposal.proposal_id)
    [refund] = provider_refunds(payments, "ord_1001")
    assert (refund["provider_reference"], refund["idempotency_key"]) == (
        operation.provider_reference,
        operation.idempotency_key,
    )
    assert operation.provider_response["status"] == "succeeded"
    assert proposal_status(db, proposal.proposal_id) == "executed"


def test_execution_is_audited(db: Session, payments: ProviderClient) -> None:
    proposal = approved(db, payments)
    operation = execute_approved_refund(db, payments, proposal.proposal_id)
    rows = db.execute(
        text(
            "SELECT event_type, actor, redacted_payload FROM support.audit_events "
            "WHERE run_id = :run AND event_type LIKE 'refund_%' ORDER BY id"
        ),
        {"run": proposal.run_id},
    ).all()
    assert [(r.event_type, r.actor) for r in rows] == [
        ("refund_reserved", "system:executor"),
        ("refund_executed", "system:executor"),
    ]
    assert rows[1].redacted_payload == {
        "operation_id": operation.operation_id,
        "proposal_id": proposal.proposal_id,
        "status": "succeeded",
        "amount_minor": 249_900,
        "error_code": None,
    }


def test_high_value_refund_after_senior_approval(db: Session, payments: ProviderClient) -> None:
    proposal = approved(db, payments, BOB, "ord_1014")
    assert execute_approved_refund(db, payments, proposal.proposal_id).status == "succeeded"


def test_executing_again_sends_nothing_new(db: Session, payments: ProviderClient) -> None:
    proposal = approved(db, payments)
    first = execute_approved_refund(db, payments, proposal.proposal_id)
    payments.http.post("/admin/faults", json={"mode": "error"})  # would fail a new call

    again = execute_approved_refund(db, payments, proposal.proposal_id)

    assert again == first
    assert len(provider_refunds(payments, "ord_1001")) == 1
    assert operations(db, "ord_1001") == ["succeeded"]


# --- No execution without a valid approval or eligibility -------------------------------


def test_pending_approval_is_not_enough(db: Session, payments: ProviderClient) -> None:
    _, _, proposal = propose(db, payments, ALICE, "ord_1001")
    result = failure(lambda: execute_approved_refund(db, payments, proposal.proposal_id))
    assert result.error_code == "APPROVAL_REQUIRED"
    assert provider_refunds(payments, "ord_1001") == []


def test_rejected_proposal_is_not_executed(db: Session, payments: ProviderClient) -> None:
    _, _, proposal = propose(db, payments, ALICE, "ord_1001")
    decide_approval(db, RIYA, proposal.approval_id, "rejected", now=NOW)
    result = failure(lambda: execute_approved_refund(db, payments, proposal.proposal_id))
    assert result.error_code == "APPROVAL_REQUIRED"
    assert operations(db, "ord_1001") == []


def test_unknown_proposal(db: Session, payments: ProviderClient) -> None:
    assert failure(lambda: execute_approved_refund(db, payments, "prp_nope")).error_code == (
        "PROPOSAL_NOT_FOUND"
    )


def test_eligibility_is_checked_again(db: Session, payments: ProviderClient) -> None:
    proposal = approved(db, payments)
    # The warehouse re-records the inspection as failed after the approval.
    db.execute(
        text("UPDATE support.returns SET inspection_status = 'failed' WHERE order_id = 'ord_1001'")
    )
    result = failure(lambda: execute_approved_refund(db, payments, proposal.proposal_id))
    assert (result.error_code, "INSPECTION_FAILED" in result.message) == ("NOT_ELIGIBLE", True)
    assert operations(db, "ord_1001") == []
    assert provider_refunds(payments, "ord_1001") == []


def test_refund_made_elsewhere_after_approval_blocks_execution(
    db: Session, payments: ProviderClient
) -> None:
    proposal = approved(db, payments)
    payments.http.post(
        "/refunds",
        json={
            "idempotency_key": "manual:1",
            "order_id": "ord_1001",
            "amount_minor": 249_900,
            "currency": "INR",
        },
    )
    result = failure(lambda: execute_approved_refund(db, payments, proposal.proposal_id))
    assert "ALREADY_REFUNDED" in result.message
    assert len(provider_refunds(payments, "ord_1001")) == 1


def test_no_reservation_while_the_provider_is_down(db: Session, payments: ProviderClient) -> None:
    proposal = approved(db, payments)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down = ProviderClient(
        "payment", httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://x")
    )
    result = failure(lambda: execute_approved_refund(db, down, proposal.proposal_id))
    assert (result.error_code, result.retryable) == ("PROVIDER_UNAVAILABLE", True)
    assert operations(db, "ord_1001") == []


# --- Provider outcomes ------------------------------------------------------------------


def test_pending_at_the_provider_is_submitted(db: Session, payments: ProviderClient) -> None:
    proposal = approved(db, payments)
    payments.http.post("/admin/faults", json={"mode": "delay"})
    operation = execute_approved_refund(db, payments, proposal.proposal_id)
    assert operation.status == "submitted"
    assert proposal_status(db, proposal.proposal_id) == "approved"  # not executed until confirmed


def test_provider_rejection_fails_and_releases_the_amount(
    db: Session, payments: ProviderClient
) -> None:
    proposal = approved(db, payments)
    payments.http.post(
        "/admin/faults",
        json={"mode": "error", "status_code": 400, "error_code": "CARD_CLOSED", "retryable": False},
    )
    operation = execute_approved_refund(db, payments, proposal.proposal_id)
    assert (operation.status, operation.provider_response["error_code"]) == (
        "failed",
        "CARD_CLOSED",
    )
    assert proposal_status(db, proposal.proposal_id) == "approved"
    # A failed operation holds nothing: the full amount is refundable again.
    decision = check_refund_eligibility(db, payments, ALICE, "ord_1001", today=TODAY)
    assert (decision.outcome, decision.refundable_balance_minor) == ("eligible", 249_900)
    # Executing again returns the failed operation; a new refund needs a new proposal.
    assert execute_approved_refund(db, payments, proposal.proposal_id) == operation
    assert provider_refunds(payments, "ord_1001") == []


def test_retryable_error_is_unknown_and_a_retry_succeeds_once(
    db: Session, payments: ProviderClient
) -> None:
    proposal = approved(db, payments)
    payments.http.post("/admin/faults", json={"mode": "error"})

    first = execute_approved_refund(db, payments, proposal.proposal_id)
    assert (first.status, first.provider_response["error_code"]) == (
        "unknown",
        "PROVIDER_UNAVAILABLE",
    )
    assert proposal_status(db, proposal.proposal_id) == "approved"

    retried = execute_approved_refund(db, payments, proposal.proposal_id)
    assert retried.status == "succeeded"
    assert retried.operation_id == first.operation_id
    assert len(provider_refunds(payments, "ord_1001")) == 1


def test_timeout_after_the_provider_accepted_is_safe_to_retry(
    db: Session, payments: ProviderClient
) -> None:
    """PDF section 12: the provider accepts the refund, the app times out before the reply."""
    proposal = approved(db, payments)
    with serve(payments_app) as url:
        with httpx.Client(base_url=url, headers=HEADERS) as admin:
            push_mock_payments(admin, TODAY)
            admin.post("/admin/faults", json={"mode": "accept_then_timeout", "delay_seconds": 1})
        slow = ProviderClient("payment", httpx.Client(base_url=url, headers=HEADERS, timeout=0.3))

        first = execute_approved_refund(db, slow, proposal.proposal_id)
        assert (first.status, first.provider_reference) == ("unknown", None)
        assert len(slow.get("/refunds", params={"order_id": "ord_1001"})) == 1  # it did happen

        retried = execute_approved_refund(db, slow, proposal.proposal_id)
        assert retried.status == "succeeded"
        refunds = slow.get("/refunds", params={"order_id": "ord_1001"})
        assert [r["provider_reference"] for r in refunds] == [retried.provider_reference]


# --- W3.5: recheck before executing only the approved action -------------------------------


def test_recheck_is_audited_before_the_reservation(db: Session, payments: ProviderClient) -> None:
    proposal = approved(db, payments)
    execute_approved_refund(db, payments, proposal.proposal_id)

    events = db.execute(
        text(
            "SELECT event_type, redacted_payload FROM support.audit_events "
            "WHERE run_id = :run ORDER BY id"
        ),
        {"run": proposal.run_id},
    ).all()
    types = [e.event_type for e in events]
    assert types.index("eligibility_rechecked") < types.index("refund_reserved")
    recheck = events[types.index("eligibility_rechecked")].redacted_payload
    assert (recheck["outcome"], recheck["amount_minor"]) == ("eligible", 249900)


def test_a_payload_that_no_longer_matches_the_approval_is_not_executed(
    db: Session, payments: ProviderClient
) -> None:
    proposal = approved(db, payments)
    # Bypass the immutability trigger (this test's transaction only) to simulate tampering.
    db.execute(text("ALTER TABLE support.action_proposals DISABLE TRIGGER USER"))
    db.execute(
        text(
            "UPDATE support.action_proposals "
            "SET payload = jsonb_set(payload, '{amount_minor}', '999900') WHERE id = :id"
        ),
        {"id": proposal.proposal_id},
    )
    db.execute(text("ALTER TABLE support.action_proposals ENABLE TRIGGER USER"))

    result = failure(lambda: execute_approved_refund(db, payments, proposal.proposal_id))

    assert result.error_code == "APPROVAL_REQUIRED"
    assert provider_refunds(payments, "ord_1001") == []
    assert operations(db, "ord_1001") == []
