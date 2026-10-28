"""Keep the external helpdesk (mock ticketing API) in step with our tickets (W3.14, Q1).

The `tickets` table is the local record, and a ticket's **status changes only after the
ticketing API confirms it** (HLD §9). Category and summary are local fields and change at
once. Every call to the helpdesk is idempotent: the external ticket is created by
`local_ticket_id`, and each update carries an `update_id` derived from its content, so a
retry sends the same update again and the helpdesk answers with the stored one.

If the helpdesk cannot be reached, only the ticket update is queued for a retry
(`ticket_sync` job); nothing else in the run is repeated for it. A newer update for the same
ticket replaces any queued older one, so a late retry can never set an outdated status.
"""

import hashlib
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SCHEMA
from app.jobs import queue
from app.services.audit import record_event
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient
from app.services.tickets import TicketView, update_ticket

JOB_TYPE = "ticket_sync"


def update_id(run_id: str, status: str | None, note: str | None) -> str:
    """Stable for the same change from the same run, different for any other change."""
    digest = hashlib.sha256(f"{status}|{note}".encode()).hexdigest()[:16]
    return f"tku:{run_id}:{digest}"


def sync_ticket(
    db: Session,
    ticketing: ProviderClient,
    ticket_id: str,
    *,
    actor: str,
    update_id: str,
    status: str | None = None,
    category: str | None = None,
    summary: str | None = None,
) -> TicketView:
    """Change a ticket: local fields now, the status once the helpdesk has confirmed it."""
    cancel_pending(db, ticket_id)
    try:
        push(db, ticketing, ticket_id, update_id=update_id, status=status, note=summary)
    except ServiceFailure as failure:
        if not failure.retryable:
            raise
        queue.enqueue(
            db,
            JOB_TYPE,
            {
                "ticket_id": ticket_id,
                "update_id": update_id,
                "status": status,
                "note": summary,
                "actor": actor,
            },
        )
        record_event(
            db,
            actor=actor,
            event_type="ticket_sync_pending",
            payload={"status": status, "error_code": failure.error_code},
            ticket_id=ticket_id,
        )
        db.commit()
        # The status waits for the helpdesk; local fields need not.
        return update_ticket(db, ticket_id, actor=actor, category=category, summary=summary)
    return update_ticket(
        db, ticket_id, actor=actor, status=status, category=category, summary=summary
    )


def retry_sync(db: Session, ticketing: ProviderClient, payload: dict[str, Any]) -> TicketView:
    """The `ticket_sync` job: send the same update again; raises while the helpdesk is down."""
    push(
        db,
        ticketing,
        payload["ticket_id"],
        update_id=payload["update_id"],
        status=payload["status"],
        note=payload["note"],
    )
    return update_ticket(db, payload["ticket_id"], actor=payload["actor"], status=payload["status"])


def push(
    db: Session,
    ticketing: ProviderClient,
    ticket_id: str,
    *,
    update_id: str,
    status: str | None,
    note: str | None,
) -> None:
    """Create the external ticket if needed, then send the update; both are idempotent."""
    ticket = db.execute(
        text(f"SELECT customer_id, external_ticket_ref FROM {SCHEMA}.tickets WHERE id = :id"),
        {"id": ticket_id},
    ).one_or_none()
    if ticket is None:
        raise ServiceFailure("TICKET_NOT_FOUND", "No ticket with this ID was found.")
    ref = ticket.external_ticket_ref
    if ref is None:
        created = ticketing.request(
            "POST",
            "/tickets",
            json={
                "local_ticket_id": ticket_id,
                "customer_id": ticket.customer_id,
                "subject": f"Support ticket {ticket_id}",
            },
        )
        ref = created["ref"]
        db.execute(
            text(f"UPDATE {SCHEMA}.tickets SET external_ticket_ref = :ref WHERE id = :id"),
            {"ref": ref, "id": ticket_id},
        )
        db.commit()
    if status is None and not note:
        return
    ticketing.request(
        "POST",
        f"/tickets/{ref}/updates",
        json={"update_id": update_id, "status": status, "note": note or None},
    )


def cancel_pending(db: Session, ticket_id: str) -> None:
    """A newer update supersedes queued retries of older ones for this ticket."""
    db.execute(
        text(
            f"UPDATE {SCHEMA}.jobs SET status = 'cancelled', finished_at = now() "
            "WHERE job_type = :type AND status = 'queued' AND payload->>'ticket_id' = :ticket"
        ),
        {"type": JOB_TYPE, "ticket": ticket_id},
    )
