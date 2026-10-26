"""Ticket creation (W1.7), reading (W1.12) and updates (W1.15).

The customer comes from authentication; an order ID from the request is only accepted after
checking, here in the service, that the order belongs to that customer.
"""

import secrets

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.schemas.tickets import TicketCreate, TicketResponse, TicketView
from app.services.audit import record_event
from app.services.errors import ServiceFailure
from app.services.orders import ensure_order_owner

CATEGORIES = ("order_status", "refund_request", "damaged_item", "out_of_scope")
STATUSES = ("open", "awaiting_customer", "awaiting_approval", "escalated", "resolved")
TICKET_COLUMNS = (
    "id, order_id, category, status, customer_message, summary, "
    "external_ticket_ref, created_at, updated_at"
)


def new_ticket_id() -> str:
    return f"tkt_{secrets.token_hex(8)}"


def create_ticket(
    db: Session, customer: AuthenticatedCustomer, request: TicketCreate
) -> TicketResponse:
    if request.order_id is not None:
        ensure_order_owner(db, customer, request.order_id)

    ticket = db.execute(
        text(
            f"INSERT INTO {SCHEMA}.tickets (id, customer_id, order_id, status, customer_message) "
            "VALUES (:id, :customer, :order, 'open', :message) "
            "RETURNING id, order_id, status, category, created_at"
        ),
        {
            "id": new_ticket_id(),
            "customer": customer.customer_id,
            "order": request.order_id,
            "message": request.message,
        },
    ).one()
    # Redacted: the message itself stays in tickets, only its length is audited.
    record_event(
        db,
        actor=f"customer:{customer.customer_id}",
        event_type="ticket_created",
        payload={"order_id": request.order_id, "message_length": len(request.message)},
        ticket_id=ticket.id,
    )
    db.commit()
    return TicketResponse.model_validate(ticket._mapping)


def get_ticket(db: Session, customer: AuthenticatedCustomer, ticket_id: str) -> TicketView:
    ticket = db.execute(
        text(
            f"SELECT {TICKET_COLUMNS} FROM {SCHEMA}.tickets "
            "WHERE id = :ticket AND customer_id = :customer"
        ),
        {"ticket": ticket_id, "customer": customer.customer_id},
    ).one_or_none()
    if ticket is None:
        # Same failure for a missing ticket and someone else's.
        raise ServiceFailure("TICKET_NOT_FOUND", "No ticket with this ID was found.")
    return TicketView.model_validate(ticket._mapping)


def update_ticket(
    db: Session,
    ticket_id: str,
    *,
    actor: str,
    status: str | None = None,
    category: str | None = None,
    summary: str | None = None,
) -> TicketView:
    """Change a ticket's status, category or summary; fields left as None keep their value.

    Called by the workflow or a reviewer, not the customer, so there is no ownership check;
    `actor` is kind:id as in the audit trail. A resolved ticket is closed and cannot change.
    """
    if status is not None and status not in STATUSES:
        raise ValueError(f"unknown ticket status {status!r}")
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"unknown ticket category {category!r}")
    ticket = db.execute(
        text(
            f"UPDATE {SCHEMA}.tickets SET status = COALESCE(:status, status), "
            "category = COALESCE(:category, category), summary = COALESCE(:summary, summary), "
            "updated_at = now() WHERE id = :ticket AND status <> 'resolved' "
            f"RETURNING {TICKET_COLUMNS}"
        ),
        {"ticket": ticket_id, "status": status, "category": category, "summary": summary},
    ).one_or_none()
    if ticket is None:
        exists = db.execute(
            text(f"SELECT 1 FROM {SCHEMA}.tickets WHERE id = :ticket"), {"ticket": ticket_id}
        ).scalar_one_or_none()
        if exists is None:
            raise ServiceFailure("TICKET_NOT_FOUND", "No ticket with this ID was found.")
        raise ServiceFailure("TICKET_CLOSED", "This ticket is resolved and cannot change.")
    # Redacted: the summary text stays in tickets, only whether it changed is audited.
    record_event(
        db,
        actor=actor,
        event_type="ticket_updated",
        payload={"status": status, "category": category, "summary_changed": summary is not None},
        ticket_id=ticket_id,
    )
    db.commit()
    return TicketView.model_validate(ticket._mapping)
