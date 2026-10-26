"""Reviewer decisions: once, before expiry, by a reviewer with the required role (W1.15)."""

from collections.abc import Iterator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.auth.reviewer import AuthenticatedReviewer
from app.services.approvals import decide_approval
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient
from app.services.runs import get_run
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from scripts.seed_data import push_mock_payments, seed
from tests.integration.test_proposals import ALICE, BOB, NOW, TODAY, propose

RIYA = AuthenticatedReviewer("auth|reviewer_riya", "reviewer")
SAM = AuthenticatedReviewer("auth|senior_sam", "senior_reviewer")


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


def statuses(db: Session, proposal_id: str) -> tuple[str, str | None]:
    row = db.execute(
        text(
            "SELECT p.status, a.decision FROM support.action_proposals p "
            "JOIN support.approvals a ON a.proposal_id = p.id WHERE p.id = :id"
        ),
        {"id": proposal_id},
    ).one()
    return row.status, row.decision


def failure(call) -> ServiceFailure:
    with pytest.raises(ServiceFailure) as error:
        call()
    return error.value


def test_approve(db: Session, payments: ProviderClient) -> None:
    run, _, proposal = propose(db, payments, ALICE, "ord_1001")
    decided_at = NOW + timedelta(minutes=5)

    approval = decide_approval(db, RIYA, proposal.approval_id, "approved", now=decided_at)

    assert (approval.decision, approval.reviewer_id, approval.reviewer_role) == (
        "approved",
        "auth|reviewer_riya",
        "reviewer",
    )
    assert approval.decided_at == decided_at
    assert statuses(db, proposal.proposal_id) == ("approved", "approved")
    assert get_run(db, ALICE, run.id).status == "running"


def test_reject_ends_the_run(db: Session, payments: ProviderClient) -> None:
    run, _, proposal = propose(db, payments, ALICE, "ord_1001")
    approval = decide_approval(db, RIYA, proposal.approval_id, "rejected", now=NOW)
    assert approval.decision == "rejected"
    assert statuses(db, proposal.proposal_id) == ("rejected", "rejected")
    finished = get_run(db, ALICE, run.id)
    assert (finished.status, finished.completed_at is not None) == ("completed", True)


def test_decision_is_audited(db: Session, payments: ProviderClient) -> None:
    run, _, proposal = propose(db, payments, ALICE, "ord_1001")
    decide_approval(db, RIYA, proposal.approval_id, "approved", now=NOW)
    event = db.execute(
        text(
            "SELECT actor, redacted_payload FROM support.audit_events "
            "WHERE run_id = :run AND event_type = 'approval_decided'"
        ),
        {"run": run.id},
    ).one()
    assert event.actor == "reviewer:auth|reviewer_riya"
    assert event.redacted_payload == {
        "approval_id": proposal.approval_id,
        "proposal_id": proposal.proposal_id,
        "decision": "approved",
        "reviewer_role": "reviewer",
    }


def test_expired_approval_cannot_be_decided(db: Session, payments: ProviderClient) -> None:
    _, _, proposal = propose(db, payments, ALICE, "ord_1001")
    late = proposal.expires_at + timedelta(seconds=1)

    result = failure(lambda: decide_approval(db, RIYA, proposal.approval_id, "approved", now=late))

    assert (result.error_code, result.retryable) == ("APPROVAL_EXPIRED", False)
    assert statuses(db, proposal.proposal_id) == ("expired", "expired")
    reviewer = db.execute(
        text("SELECT reviewer_id FROM support.approvals WHERE id = :id"),
        {"id": proposal.approval_id},
    ).scalar_one()
    assert reviewer is None  # the system expired it, no reviewer decided


def test_decision_at_the_expiry_moment_counts(db: Session, payments: ProviderClient) -> None:
    _, _, proposal = propose(db, payments, ALICE, "ord_1001")
    approval = decide_approval(db, RIYA, proposal.approval_id, "approved", now=proposal.expires_at)
    assert approval.decision == "approved"


def test_standard_reviewer_cannot_decide_a_high_value_refund(
    db: Session, payments: ProviderClient
) -> None:
    _, _, proposal = propose(db, payments, BOB, "ord_1014")
    for decision in ("approved", "rejected"):
        result = failure(
            lambda d=decision: decide_approval(db, RIYA, proposal.approval_id, d, now=NOW)
        )
        assert result.error_code == "REVIEWER_NOT_PERMITTED"
    assert statuses(db, proposal.proposal_id) == ("pending_approval", None)

    approval = decide_approval(db, SAM, proposal.approval_id, "approved", now=NOW)
    assert (approval.decision, approval.reviewer_role) == ("approved", "senior_reviewer")


def test_senior_reviewer_can_decide_a_standard_refund(
    db: Session, payments: ProviderClient
) -> None:
    _, _, proposal = propose(db, payments, ALICE, "ord_1001")
    assert (
        decide_approval(db, SAM, proposal.approval_id, "approved", now=NOW).decision == "approved"
    )


@pytest.mark.parametrize("second", ["approved", "rejected"])
def test_an_approval_is_decided_once(db: Session, payments: ProviderClient, second: str) -> None:
    _, _, proposal = propose(db, payments, ALICE, "ord_1001")
    decide_approval(db, RIYA, proposal.approval_id, "approved", now=NOW)
    result = failure(lambda: decide_approval(db, SAM, proposal.approval_id, second, now=NOW))
    assert result.error_code == "APPROVAL_ALREADY_DECIDED"
    assert statuses(db, proposal.proposal_id) == ("approved", "approved")


def test_unknown_approval(db: Session) -> None:
    result = failure(lambda: decide_approval(db, RIYA, "apr_nope", "approved", now=NOW))
    assert result.error_code == "APPROVAL_NOT_FOUND"


def test_unknown_decision_is_a_bug(db: Session, payments: ProviderClient) -> None:
    _, _, proposal = propose(db, payments, ALICE, "ord_1001")
    with pytest.raises(ValueError):
        decide_approval(db, RIYA, proposal.approval_id, "maybe", now=NOW)
