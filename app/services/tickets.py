"""Ticket creation (W1.7).

The customer comes from authentication; an order ID from the request is only accepted after
checking, here in the service, that the order belongs to that customer.
"""

import json
import secrets

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.schemas.tickets import TicketCreate, TicketResponse
from app.services.errors import ServiceFailure


def new_ticket_id() -> str:
    return f"tkt_{secrets.token_hex(8)}"


def create_ticket(
    db: Session, customer: AuthenticatedCustomer, request: TicketCreate
) -> TicketResponse:
    if request.order_id is not None:
        owned = db.execute(
            text(f"SELECT 1 FROM {SCHEMA}.orders WHERE id = :order AND customer_id = :customer"),
            {"order": request.order_id, "customer": customer.customer_id},
        ).first()
        if owned is None:
            # Same failure whether the order is missing or someone else's, so order IDs
            # cannot be probed.
            raise ServiceFailure("ORDER_NOT_FOUND", "No order with this ID was found.")

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
