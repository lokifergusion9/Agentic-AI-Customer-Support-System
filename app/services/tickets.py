"""Ticket creation (W1.7) and reading (W1.12).

The customer comes from authentication; an order ID from the request is only accepted after
checking, here in the service, that the order belongs to that customer.
"""

import json
import secrets

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.schemas.tickets import TicketCreate, TicketResponse, TicketView
from app.services.errors import ServiceFailure
from app.services.orders import ensure_order_owner


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
    db.execute(
        text(
            f"INSERT INTO {SCHEMA}.audit_events (ticket_id, actor, event_type, redacted_payload) "
            "VALUES (:ticket, :actor, 'ticket_created', CAST(:payload AS jsonb))"
        ),
        {
            "ticket": ticket.id,
            "actor": f"customer:{customer.customer_id}",
            "payload": json.dumps(
                {"order_id": request.order_id, "message_length": len(request.message)}
            ),
        },
    )
    db.commit()
    return TicketResponse.model_validate(ticket._mapping)


def get_ticket(db: Session, customer: AuthenticatedCustomer, ticket_id: str) -> TicketView:
    ticket = db.execute(
        text(
            "SELECT id, order_id, category, status, customer_message, summary, "
            f"external_ticket_ref, created_at, updated_at FROM {SCHEMA}.tickets "
            "WHERE id = :ticket AND customer_id = :customer"
        ),
        {"ticket": ticket_id, "customer": customer.customer_id},
    ).one_or_none()
    if ticket is None:
        # Same failure for a missing ticket and someone else's.
        raise ServiceFailure("TICKET_NOT_FOUND", "No ticket with this ID was found.")
    return TicketView.model_validate(ticket._mapping)
