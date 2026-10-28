"""Workflow runs (W1.15 basic). A run is one attempt to resolve a ticket.

The LangGraph workflow (W2) drives runs; this module only records them. A ticket has at most
one unfinished run (enforced by the database).
"""

import json
import secrets
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.services.audit import record_event
from app.services.errors import ServiceFailure
from app.services.tickets import get_ticket

TERMINAL_STATUSES = ("completed", "escalated", "failed")
STATUSES = ("running", "waiting_for_customer", "waiting_for_approval", *TERMINAL_STATUSES)


class RunView(BaseModel):
    id: str
    ticket_id: str
    thread_id: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    # Final response, route, category, proposal and citations (W2.14); None while running.
    output: dict[str, Any] | None = None


RUN_COLUMNS = "id, ticket_id, thread_id, status, started_at, completed_at, output"


def run_not_found() -> ServiceFailure:
    return ServiceFailure("RUN_NOT_FOUND", "No run with this ID was found.")


def start_run(db: Session, customer: AuthenticatedCustomer, ticket_id: str) -> RunView:
    get_ticket(db, customer, ticket_id)  # TICKET_NOT_FOUND unless it is the customer's
    try:
        with db.begin_nested():
            run = db.execute(
                text(
                    f"INSERT INTO {SCHEMA}.workflow_runs (id, ticket_id, thread_id, status) "
                    f"VALUES (:id, :ticket, :thread, 'running') RETURNING {RUN_COLUMNS}"
                ),
                {
                    "id": f"run_{secrets.token_hex(8)}",
                    "ticket": ticket_id,
                    "thread": f"thr_{secrets.token_hex(8)}",
                },
            ).one()
    except IntegrityError:
        raise ServiceFailure(
            "RUN_ALREADY_ACTIVE", "This ticket already has an unfinished run."
        ) from None
    record_event(
        db,
        actor=f"customer:{customer.customer_id}",
        event_type="run_started",
        payload={"ticket_id": ticket_id},
        ticket_id=ticket_id,
        run_id=run.id,
    )
    db.commit()
    return RunView.model_validate(run._mapping)


def get_run(db: Session, customer: AuthenticatedCustomer, run_id: str) -> RunView:
    run = db.execute(
        text(
            f"SELECT r.{RUN_COLUMNS.replace(', ', ', r.')} FROM {SCHEMA}.workflow_runs r "
            f"JOIN {SCHEMA}.tickets t ON t.id = r.ticket_id "
            "WHERE r.id = :run AND t.customer_id = :customer"
        ),
        {"run": run_id, "customer": customer.customer_id},
    ).one_or_none()
    if run is None:
        raise run_not_found()
    return RunView.model_validate(run._mapping)


def get_run_for_reviewer(db: Session, run_id: str) -> RunView:
    """Reviewers may inspect any run (they decide its approvals)."""
    run = db.execute(
        text(f"SELECT {RUN_COLUMNS} FROM {SCHEMA}.workflow_runs WHERE id = :run"), {"run": run_id}
    ).one_or_none()
    if run is None:
        raise run_not_found()
    return RunView.model_validate(run._mapping)


def list_run_events(db: Session, run_id: str) -> list[dict[str, Any]]:
    """The run's redacted audit timeline, oldest first. Access is checked by the caller."""
    rows = db.execute(
        text(
            "SELECT id, sequence_number, created_at, actor, event_type, "
            f"redacted_payload AS payload FROM {SCHEMA}.audit_events "
            "WHERE run_id = :run ORDER BY sequence_number"
        ),
        {"run": run_id},
    ).all()
    return [dict(row._mapping) for row in rows]


def set_run_output(db: Session, run_id: str, output: dict[str, Any]) -> None:
    """Store the run's latest result in the caller's transaction."""
    db.execute(
        text(f"UPDATE {SCHEMA}.workflow_runs SET output = CAST(:output AS jsonb) WHERE id = :run"),
        {"run": run_id, "output": json.dumps(output, default=str)},
    )


def set_run_status(db: Session, run_id: str, status: str, *, actor: str) -> RunView:
    """Move a run to `status` in the caller's transaction; finished runs cannot move."""
    if status not in STATUSES:
        raise ValueError(f"unknown run status {status!r}")
    run = db.execute(
        text(
            f"UPDATE {SCHEMA}.workflow_runs SET status = :status, "
            "completed_at = CASE WHEN :status IN ('completed', 'escalated', 'failed') "
            "THEN now() END "
            "WHERE id = :run AND status NOT IN ('completed', 'escalated', 'failed') "
            f"RETURNING {RUN_COLUMNS}"
        ),
        {"run": run_id, "status": status},
    ).one_or_none()
    if run is None:
        raise ServiceFailure("RUN_NOT_ACTIVE", "The run is finished or does not exist.")
    record_event(
        db,
        actor=actor,
        event_type="run_status_changed",
        payload={"status": status},
        ticket_id=run.ticket_id,
        run_id=run_id,
    )
    return RunView.model_validate(run._mapping)
