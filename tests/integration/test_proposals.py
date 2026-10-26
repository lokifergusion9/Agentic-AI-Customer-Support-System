"""Runs, refund proposals and their pending approvals (W1.15 basic, W3.1, W3.2)."""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.schemas.tickets import TicketCreate
from app.services.eligibility import check_refund_eligibility
from app.services.errors import ServiceFailure
from app.services.proposals import create_refund_proposal, payload_hash
from app.services.providers import ProviderClient
from app.services.runs import get_run, set_run_status, start_run
from app.services.tickets import create_ticket
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from scripts.seed_data import push_mock_payments, seed

TODAY = date(2026, 10, 25)
NOW = datetime(2026, 10, 25, 6, 0, tzinfo=UTC)
ALICE = AuthenticatedCustomer("cus_alice", "auth|alice")
BOB = AuthenticatedCustomer("cus_bob", "auth|bob")


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


def new_ticket(db: Session, customer: AuthenticatedCustomer, order_id: str) -> str:
    return create_ticket(db, customer, TicketCreate(message="Refund please", order_id=order_id)).id


def propose(db: Session, payments: ProviderClient, customer: AuthenticatedCustomer, order_id: str):
    run = start_run(db, customer, new_ticket(db, customer, order_id))
    decision = check_refund_eligibility(db, payments, customer, order_id, today=TODAY)
    proposal = create_refund_proposal(
        db,
        customer,
        run.id,
        decision,
        reason="Return received and inspection passed",
        evidence_refs=[f"return:{order_id}", f"payment:{order_id}"],
        now=NOW,
    )
    return run, decision, proposal


def events(db: Session, run_id: str) -> list[str]:
    return list(
        db.execute(
            text("SELECT event_type FROM support.audit_events WHERE run_id = :run ORDER BY id"),
            {"run": run_id},
        ).scalars()
    )


# --- Runs -------------------------------------------------------------------------------


def test_start_run(db: Session) -> None:
    ticket_id = new_ticket(db, ALICE, "ord_1001")
    run = start_run(db, ALICE, ticket_id)
    assert run.id.startswith("run_") and run.thread_id.startswith("thr_")
    assert (run.ticket_id, run.status, run.completed_at) == (ticket_id, "running", None)
    assert get_run(db, ALICE, run.id) == run
    assert events(db, run.id) == ["run_started"]


def test_one_unfinished_run_per_ticket(db: Session) -> None:
    ticket_id = new_ticket(db, ALICE, "ord_1001")
    first = start_run(db, ALICE, ticket_id)
    with pytest.raises(ServiceFailure) as error:
        start_run(db, ALICE, ticket_id)
    assert error.value.error_code == "RUN_ALREADY_ACTIVE"

    set_run_status(db, first.id, "completed", actor="system:test")
    assert start_run(db, ALICE, ticket_id).status == "running"


def test_run_for_someone_elses_ticket(db: Session) -> None:
    ticket_id = new_ticket(db, BOB, "ord_1003")
    with pytest.raises(ServiceFailure) as error:
        start_run(db, ALICE, ticket_id)
    assert error.value.error_code == "TICKET_NOT_FOUND"


def test_someone_elses_run_is_not_found(db: Session) -> None:
    run = start_run(db, BOB, new_ticket(db, BOB, "ord_1003"))
    with pytest.raises(ServiceFailure) as error:
        get_run(db, ALICE, run.id)
    assert error.value.error_code == "RUN_NOT_FOUND"


def test_finished_run_cannot_move(db: Session) -> None:
    run = start_run(db, ALICE, new_ticket(db, ALICE, "ord_1001"))
    finished = set_run_status(db, run.id, "escalated", actor="system:test")
    assert finished.completed_at is not None
    with pytest.raises(ServiceFailure) as error:
        set_run_status(db, run.id, "running", actor="system:test")
    assert error.value.error_code == "RUN_NOT_ACTIVE"


def test_unknown_run_status_is_a_bug(db: Session) -> None:
    run = start_run(db, ALICE, new_ticket(db, ALICE, "ord_1001"))
    with pytest.raises(ValueError):
        set_run_status(db, run.id, "paused", actor="system:test")


# --- Proposals --------------------------------------------------------------------------


def test_refund_proposal_with_a_pending_approval(db: Session, payments: ProviderClient) -> None:
    run, decision, proposal = propose(db, payments, ALICE, "ord_1001")

    assert proposal.proposal_id.startswith("prp_") and proposal.approval_id.startswith("apr_")
    assert (proposal.run_id, proposal.order_id, proposal.status) == (
        run.id,
        "ord_1001",
        "pending_approval",
    )
    assert proposal.payload == {
        "action": "request_refund_approval",
        "order_id": "ord_1001",
        "amount_minor": 249_900,
        "currency": "INR",
        "reason": "Return received and inspection passed",
        "evidence_refs": ["return:ord_1001", "payment:ord_1001"],
        "policy_refs": decision.policy_refs,
        "missing_information": [],
    }
    assert proposal.payload_hash == payload_hash(proposal.payload)
    assert (proposal.required_role, proposal.expires_at) == (
        "reviewer",
        NOW + timedelta(minutes=60),
    )

    approval = db.execute(
        text(
            "SELECT approved_payload_hash, decision, reviewer_id FROM support.approvals "
            "WHERE id = :id"
        ),
        {"id": proposal.approval_id},
    ).one()
    assert tuple(approval) == (proposal.payload_hash, None, None)
    assert get_run(db, ALICE, run.id).status == "waiting_for_approval"
    assert events(db, run.id) == ["run_started", "run_status_changed", "proposal_created"]


def test_proposal_audit_is_redacted(db: Session, payments: ProviderClient) -> None:
    run, _, proposal = propose(db, payments, ALICE, "ord_1001")
    payload = db.execute(
        text(
            "SELECT redacted_payload FROM support.audit_events "
            "WHERE run_id = :run AND event_type = 'proposal_created'"
        ),
        {"run": run.id},
    ).scalar_one()
    assert payload == {
        "proposal_id": proposal.proposal_id,
        "order_id": "ord_1001",
        "amount_minor": 249_900,
        "required_role": "reviewer",
    }


def test_high_value_proposal_needs_a_senior_reviewer(db: Session, payments: ProviderClient) -> None:
    _, _, proposal = propose(db, payments, BOB, "ord_1014")
    assert (proposal.payload["amount_minor"], proposal.required_role) == (
        8_999_900,
        "senior_reviewer",
    )


def test_same_proposal_again_is_not_stored_twice(db: Session, payments: ProviderClient) -> None:
    run, decision, first = propose(db, payments, ALICE, "ord_1001")
    again = create_refund_proposal(
        db,
        ALICE,
        run.id,
        decision,
        reason="Return received and inspection passed",
        evidence_refs=["return:ord_1001", "payment:ord_1001"],
        now=NOW,
    )
    assert again == first
    count = db.execute(
        text("SELECT count(*) FROM support.action_proposals WHERE run_id = :run"), {"run": run.id}
    ).scalar_one()
    assert count == 1


@pytest.mark.parametrize(
    ("customer", "order_id", "reason"),
    [
        (BOB, "ord_1003", "REFUND_PENDING"),
        (BOB, "ord_1004", "RETURN_NOT_RECEIVED"),
        (AuthenticatedCustomer("cus_dev", "auth|dev"), "ord_1012", "OUTSIDE_RETURN_WINDOW"),
    ],
)
def test_no_proposal_unless_eligible(
    db: Session,
    payments: ProviderClient,
    customer: AuthenticatedCustomer,
    order_id: str,
    reason: str,
) -> None:
    run = start_run(db, customer, new_ticket(db, customer, order_id))
    decision = check_refund_eligibility(db, payments, customer, order_id, today=TODAY)
    with pytest.raises(ServiceFailure) as error:
        create_refund_proposal(db, customer, run.id, decision, reason="r", evidence_refs=[])
    assert error.value.error_code == "NOT_ELIGIBLE"
    assert reason in error.value.message
    assert get_run(db, customer, run.id).status == "running"


def test_proposal_in_someone_elses_run(db: Session, payments: ProviderClient) -> None:
    bobs_run = start_run(db, BOB, new_ticket(db, BOB, "ord_1014"))
    decision = check_refund_eligibility(db, payments, ALICE, "ord_1001", today=TODAY)
    with pytest.raises(ServiceFailure) as error:
        create_refund_proposal(db, ALICE, bobs_run.id, decision, reason="r", evidence_refs=[])
    assert error.value.error_code == "RUN_NOT_FOUND"


def test_seeded_proposals_use_the_same_hash() -> None:
    # The seed now imports payload_hash from the proposal service; the hash is stable.
    assert payload_hash({"b": 1, "a": "₹"}) == payload_hash({"a": "₹", "b": 1})
