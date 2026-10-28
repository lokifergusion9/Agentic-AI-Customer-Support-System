"""Ticket updates reach the helpdesk; a failed one is retried alone (W3.14)."""

from collections.abc import Iterator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.approvals import decide_approval
from app.services.helpdesk import sync_ticket, update_id
from app.services.providers import ProviderClient
from app.services.tickets import get_ticket
from app.workflows.recovery import recover
from app.workflows.runner import resume_after_decision
from mock_services.ticketing.main import app as ticketing_app
from mock_services.ticketing.main import settings as ticketing_settings
from scripts.seed_data import push_mock_ticketing
from tests.integration import test_workflow as workflow
from tests.integration.test_approvals import RIYA
from tests.integration.test_proposals import ALICE, TODAY
from tests.integration.test_workflow import HEADPHONES, headphones_scenario, open_ticket, refund_ids

# The workflow tests' fixtures: seeded database, mock providers, scripted services.
db = workflow.db
providers = workflow.providers
make_services = workflow.make_services


@pytest.fixture
def ticketing() -> Iterator[ProviderClient]:
    headers = {"X-API-Key": ticketing_settings.mock_ticketing_api_key}
    with TestClient(ticketing_app, headers=headers) as http:
        push_mock_ticketing(http, TODAY)
        yield ProviderClient("ticketing", http)


def external(ticketing: ProviderClient, db: Session, ticket_id: str) -> dict:
    ref = db.execute(
        text("SELECT external_ticket_ref FROM support.tickets WHERE id = :id"), {"id": ticket_id}
    ).scalar_one()
    return ticketing.get(f"/tickets/{ref}")


def pending_syncs(db: Session) -> list[tuple[str, str]]:
    return [
        (row.payload["status"], row.status)
        for row in db.execute(
            text(
                "SELECT payload, status FROM support.jobs WHERE job_type = 'ticket_sync' "
                "ORDER BY created_at"
            )
        )
    ]


def sync(db: Session, ticketing: ProviderClient, ticket_id: str, status: str, note: str = "n"):
    return sync_ticket(
        db,
        ticketing,
        ticket_id,
        actor="system:workflow",
        update_id=update_id("run_x", status, note),
        status=status,
        summary=note,
    )


def test_status_changes_after_the_helpdesk_confirms(db: Session, ticketing: ProviderClient) -> None:
    ticket_id = open_ticket(db, HEADPHONES, "ord_1001")

    view = sync(db, ticketing, ticket_id, "awaiting_approval")

    assert view.status == "awaiting_approval"
    remote = external(ticketing, db, ticket_id)
    assert (remote["local_ticket_id"], remote["status"]) == (ticket_id, "awaiting_approval")

    # A retry of the same change is a replay, not a second update.
    sync(db, ticketing, ticket_id, "awaiting_approval")
    assert len(external(ticketing, db, ticket_id)["notes"]) == 1


def test_helpdesk_down_keeps_the_status_and_queues_only_the_update(
    db: Session, ticketing: ProviderClient
) -> None:
    ticket_id = open_ticket(db, HEADPHONES, "ord_1001")
    ticketing.http.post("/admin/faults", json={"mode": "error"})  # 503 on the next write

    view = sync(db, ticketing, ticket_id, "awaiting_approval", note="Refund awaits review")

    assert view.status == "open"  # not confirmed, so unchanged
    assert view.summary == "Refund awaits review"  # local field, changed at once
    assert pending_syncs(db) == [("awaiting_approval", "queued")]


def test_a_newer_update_replaces_a_queued_older_one(db: Session, ticketing: ProviderClient) -> None:
    ticket_id = open_ticket(db, HEADPHONES, "ord_1001")
    ticketing.http.post("/admin/faults", json={"mode": "error"})
    sync(db, ticketing, ticket_id, "awaiting_approval")

    sync(db, ticketing, ticket_id, "escalated")

    assert pending_syncs(db) == [("awaiting_approval", "cancelled")]
    assert get_ticket(db, ALICE, ticket_id).status == "escalated"


def test_refund_paid_but_ticket_update_failed_retries_only_the_ticket(
    db: Session, make_services, providers, ticketing: ProviderClient
) -> None:
    """HLD §5b: the refund succeeds, the ticket update fails, only the update is retried."""
    payments, _ = providers
    services = replace(
        make_services(headphones_scenario(*refund_ids(db, payments))), ticketing=ticketing
    )
    run = workflow.start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)
    assert get_ticket(db, ALICE, run.ticket_id).status == "awaiting_approval"
    decide_approval(db, RIYA, run.output["approval_id"], "approved")
    ticketing.http.post("/admin/faults", json={"mode": "error"})  # the final update fails

    done = resume_after_decision(db, run.output["approval_id"], services)

    assert (done.status, done.output["route"]) == ("completed", "refunded")
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1
    assert get_ticket(db, ALICE, run.ticket_id).status == "awaiting_approval"  # not confirmed
    assert pending_syncs(db) == [("resolved", "queued")]

    report = recover(db, services)

    assert report.ticket_syncs == {run.ticket_id: "synced"}
    assert get_ticket(db, ALICE, run.ticket_id).status == "resolved"
    assert external(ticketing, db, run.ticket_id)["status"] == "resolved"
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1  # the refund was not redone
    assert pending_syncs(db) == [("resolved", "completed")]
