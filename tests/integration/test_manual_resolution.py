"""Week 1 gate: resolve sample tickets through the services alone, no LLM (W1.15).

Refund: create a ticket, check eligibility, create a proposal, approve it, execute the mock
refund, update the ticket. Order status: create a ticket, read the shipment, update the ticket.
The helpdesk copy of the ticket is synced in W3.14; here the local record is updated.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.schemas.tickets import TicketCreate
from app.services.approvals import decide_approval
from app.services.eligibility import check_refund_eligibility
from app.services.proposals import create_refund_proposal
from app.services.providers import ProviderClient
from app.services.refunds import execute_approved_refund
from app.services.runs import get_run, set_run_status, start_run
from app.services.shipping import get_shipment_status
from app.services.tickets import create_ticket, get_ticket, update_ticket
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from scripts.seed_data import push_mock_payments, push_mock_shipping, seed
from tests.integration.test_approvals import RIYA
from tests.integration.test_proposals import ALICE, NOW, TODAY

ACTOR = "system:manual"


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


@pytest.fixture
def shipping() -> Iterator[ProviderClient]:
    headers = {"X-API-Key": shipping_settings.mock_shipping_api_key}
    with TestClient(shipping_app, headers=headers) as http:
        push_mock_shipping(http, TODAY)
        yield ProviderClient("shipping", http)


def events(db: Session, ticket_id: str) -> list[str]:
    return list(
        db.execute(
            text(
                "SELECT event_type FROM support.audit_events WHERE ticket_id = :ticket ORDER BY id"
            ),
            {"ticket": ticket_id},
        ).scalars()
    )


def test_refund_ticket_resolved_manually(db: Session, payments: ProviderClient) -> None:
    # 1. The customer opens a ticket about their returned headphones.
    ticket = create_ticket(
        db,
        ALICE,
        TicketCreate(
            message="I returned my headphones last week, but I still haven't received my refund.",
            order_id="ord_1001",
        ),
    )
    run = start_run(db, ALICE, ticket.id)

    # 2. Eligibility is decided in code.
    decision = check_refund_eligibility(db, payments, ALICE, "ord_1001", today=TODAY)
    assert (decision.outcome, decision.amount_minor, decision.reviewer_role) == (
        "eligible",
        249_900,
        "reviewer",
    )

    # 3. A proposal is stored and the run waits for a reviewer.
    proposal = create_refund_proposal(
        db,
        ALICE,
        run.id,
        decision,
        reason="Return received and inspection passed",
        evidence_refs=["return:ord_1001", "payment:ord_1001"],
        now=NOW,
    )
    update_ticket(
        db,
        ticket.id,
        actor=ACTOR,
        status="awaiting_approval",
        category="refund_request",
        summary="Refund of 249900 paise proposed for ord_1001.",
    )
    assert get_run(db, ALICE, run.id).status == "waiting_for_approval"

    # 4. A reviewer approves it.
    approval = decide_approval(db, RIYA, proposal.approval_id, "approved", now=NOW)
    assert (approval.decision, approval.reviewer_id) == ("approved", RIYA.reviewer_id)
    assert get_run(db, ALICE, run.id).status == "running"

    # 5. The approved refund is executed once at the mock provider.
    operation = execute_approved_refund(db, payments, proposal.proposal_id)
    assert (operation.status, operation.amount_minor) == ("succeeded", 249_900)
    [refund] = payments.get("/refunds", params={"order_id": "ord_1001"})
    assert (refund["status"], refund["provider_reference"]) == (
        "succeeded",
        operation.provider_reference,
    )

    # 6. The ticket reflects the verified provider result, and the run ends.
    set_run_status(db, run.id, "completed", actor=ACTOR)
    resolved = update_ticket(
        db,
        ticket.id,
        actor=ACTOR,
        status="resolved",
        summary=f"Refund of 249900 paise succeeded ({operation.provider_reference}).",
    )

    assert get_ticket(db, ALICE, ticket.id) == resolved
    assert (resolved.status, resolved.category) == ("resolved", "refund_request")
    finished = get_run(db, ALICE, run.id)
    assert (finished.status, finished.completed_at is not None) == ("completed", True)
    # Nothing more is refundable for this order.
    after = check_refund_eligibility(db, payments, ALICE, "ord_1001", today=TODAY)
    assert (after.outcome, after.refundable_balance_minor) == ("blocked", 0)
    assert events(db, ticket.id) == [
        "ticket_created",
        "run_started",
        "run_status_changed",  # waiting_for_approval
        "proposal_created",
        "ticket_updated",
        "approval_decided",
        "run_status_changed",  # running
        "refund_reserved",
        "refund_executed",
        "run_status_changed",  # completed
        "ticket_updated",
    ]


def test_order_status_ticket_resolved_manually(db: Session, shipping: ProviderClient) -> None:
    # 1. The customer asks where their delayed power bank is.
    ticket = create_ticket(
        db, ALICE, TicketCreate(message="Where is my power bank?", order_id="ord_1008")
    )
    run = start_run(db, ALICE, ticket.id)

    # 2. The shipment is read from the courier, after the ownership check.
    shipment = get_shipment_status(db, shipping, ALICE, "ord_1008")
    assert (shipment.status, shipment.delivered_at) == ("in_transit", None)
    assert shipment.estimated_delivery < TODAY  # running late

    # 3. The ticket is updated with the latest available information.
    set_run_status(db, run.id, "completed", actor=ACTOR)
    resolved = update_ticket(
        db,
        ticket.id,
        actor=ACTOR,
        status="resolved",
        category="order_status",
        summary=(
            f"Shipment {shipment.tracking_id} is {shipment.status}, estimated "
            f"{shipment.estimated_delivery.isoformat()}, last updated "
            f"{shipment.last_updated_at.isoformat()}."
        ),
    )

    assert get_ticket(db, ALICE, ticket.id) == resolved
    assert (resolved.status, resolved.category) == ("resolved", "order_status")
    assert shipment.tracking_id in (resolved.summary or "")
    assert get_run(db, ALICE, run.id).status == "completed"
    assert events(db, ticket.id) == [
        "ticket_created",
        "run_started",
        "run_status_changed",  # completed
        "ticket_updated",
    ]
