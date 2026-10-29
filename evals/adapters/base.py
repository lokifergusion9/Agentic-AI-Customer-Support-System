"""The adapter interface every agent under test implements (W4.6, E§10).

An adapter gets the agent view of a task (what the customer sends) and an AttemptContext (the
environment's clients and the hidden helpers, such as the reviewer simulator), runs one
attempt, and returns the final response and why it stopped. It never sees the evaluator view.
"""

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.policies.embeddings import Embedder
from app.schemas.tickets import TicketCreate
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient
from app.services.tickets import create_ticket
from evals.environment import Environment
from evals.reviewer_simulator import ReviewerSimulator
from evals.task_format import AgentInput, AgentView, Limits

ORDER_ID_IN_TEXT = re.compile(r"\bord_[A-Za-z0-9]+\b")


@dataclass
class AgentResult:
    final_response: str
    stop_reason: str
    metadata: dict[str, Any]


@dataclass
class AttemptContext:
    db: Session
    environment: Environment
    payments: ProviderClient
    shipping: ProviderClient
    embedder: Embedder
    reviewer: ReviewerSimulator  # decides approval waits; the agent never sees its behaviour
    limits: Limits
    agent_config: dict[str, Any]  # the eval.agent_configs row
    ticketing: ProviderClient | None = None
    # Set to stop the attempt: no new tool call starts after it is set (W4.10).
    cancel: threading.Event = field(default_factory=threading.Event)
    sleep: Callable[[float], None] = time.sleep


class AgentAdapter(Protocol):
    async def run(self, task_input: AgentView, context: AttemptContext) -> AgentResult: ...


class AttemptCancelled(Exception):
    pass


def resolve_customer(db: Session, subject: str) -> AuthenticatedCustomer:
    """The customer a JWT with this `sub` authenticates as, the way W1.5 resolves it."""
    customer_id = db.execute(
        text(f"SELECT id FROM {SCHEMA}.customers WHERE external_auth_subject = :subject"),
        {"subject": subject},
    ).scalar_one_or_none()
    if customer_id is None:
        raise ServiceFailure("CUSTOMER_NOT_FOUND", "The task's customer is not in the fixture.")
    return AuthenticatedCustomer(customer_id=customer_id, subject=subject)


def open_ticket(
    db: Session, customer: AuthenticatedCustomer, agent_input: AgentInput
) -> tuple[str, bool]:
    """Create the customer's ticket as POST /tickets would. Returns (ticket ID, order refused).

    POST /tickets refuses an order the customer does not own. A real customer would then
    write in without the order field, so the ticket is opened without it and the order ID
    they gave stays in their message: the agent still has to refuse it (access-boundary tasks).
    """
    order_id = agent_input.order_id
    try:
        ticket = create_ticket(
            db, customer, TicketCreate(message=agent_input.user_message, order_id=order_id)
        )
        return ticket.id, False
    except ServiceFailure as failure:
        if order_id is None or failure.error_code != "ORDER_NOT_FOUND":
            raise
        db.rollback()
    message = agent_input.user_message
    if order_id not in message:
        message = f"{message}\nOrder ID: {order_id}"
    return create_ticket(db, customer, TicketCreate(message=message, order_id=None)).id, True


def order_in_message(agent_input: AgentInput) -> str | None:
    if agent_input.order_id:
        return agent_input.order_id
    match = ORDER_ID_IN_TEXT.search(agent_input.user_message)
    return match.group(0) if match else None
