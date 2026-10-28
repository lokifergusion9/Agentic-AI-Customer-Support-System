"""How a request becomes work: inline in the request, or a job for the worker (W3.13, Q7).

`RUN_EXECUTION=inline` (the prototype, and tests) drives the graph inside the request.
`RUN_EXECUTION=worker` records the request, queues a `workflow_run` job and returns at once
(the API answers 202 Accepted); the worker process (`app/jobs/worker.py`) drives the run and
clients poll `GET /runs/{run_id}`. The API then needs no model credentials at all.
"""

from typing import Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.agents.config import AgentConfig
from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.jobs import queue
from app.services.errors import ServiceFailure
from app.services.runs import RunView, get_run_for_reviewer
from app.workflows.runner import (
    WorkflowServices,
    accept_message,
    begin_resolution,
    resume_after_decision,
    resume_with_message,
    start_resolution,
)

JOB_TYPE = "workflow_run"


class Dispatcher(Protocol):
    # True when the work is queued, not done: the API answers 202 Accepted.
    accepted: bool

    def start(self, db: Session, customer: AuthenticatedCustomer, ticket_id: str) -> RunView: ...

    def message(
        self, db: Session, customer: AuthenticatedCustomer, run_id: str, message: str
    ) -> RunView: ...

    def decision(self, db: Session, approval_id: str) -> RunView: ...


class InlineDispatcher:
    accepted = False

    def __init__(self, services: WorkflowServices) -> None:
        self.services = services

    def start(self, db: Session, customer: AuthenticatedCustomer, ticket_id: str) -> RunView:
        return start_resolution(db, customer, ticket_id, self.services)

    def message(
        self, db: Session, customer: AuthenticatedCustomer, run_id: str, message: str
    ) -> RunView:
        return resume_with_message(db, customer, run_id, message, self.services)

    def decision(self, db: Session, approval_id: str) -> RunView:
        return resume_after_decision(db, approval_id, self.services)


class WorkerDispatcher:
    accepted = True

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def start(self, db: Session, customer: AuthenticatedCustomer, ticket_id: str) -> RunView:
        run = begin_resolution(db, customer, ticket_id, self.config)
        enqueue_drive(db, run.id)
        db.commit()
        return run

    def message(
        self, db: Session, customer: AuthenticatedCustomer, run_id: str, message: str
    ) -> RunView:
        run = accept_message(db, customer, run_id, message)
        enqueue_message(db, run_id, message)
        db.commit()
        return run

    def decision(self, db: Session, approval_id: str) -> RunView:
        run_id = enqueue_decision(db, approval_id)
        db.commit()
        return get_run_for_reviewer(db, run_id)


def enqueue_drive(db: Session, run_id: str) -> str | None:
    """Drive the run from its checkpoint; at most one such job is queued per run."""
    return queue.enqueue(
        db, JOB_TYPE, {"action": "drive", "run_id": run_id}, dedupe_key=f"drive:{run_id}"
    )


def enqueue_message(db: Session, run_id: str, message: str) -> str | None:
    # The reply travels in the job, as it does in the checkpoint; the audit trail has its length.
    return queue.enqueue(db, JOB_TYPE, {"action": "message", "run_id": run_id, "message": message})


def enqueue_decision(db: Session, approval_id: str) -> str:
    """Queue the resume after a stored decision; returns the run's ID."""
    run_id = db.execute(
        text(
            f"SELECT p.run_id FROM {SCHEMA}.approvals a "
            f"JOIN {SCHEMA}.action_proposals p ON p.id = a.proposal_id WHERE a.id = :approval"
        ),
        {"approval": approval_id},
    ).scalar_one_or_none()
    if run_id is None:
        raise ServiceFailure("APPROVAL_NOT_FOUND", "No approval with this ID was found.")
    queue.enqueue(
        db, JOB_TYPE, {"action": "decision", "run_id": run_id, "approval_id": approval_id}
    )
    return run_id


def run_has_active_job(db: Session, run_id: str) -> bool:
    return (
        db.execute(
            text(
                f"SELECT 1 FROM {SCHEMA}.jobs WHERE job_type = :type "
                "AND payload->>'run_id' = :run AND status IN ('queued', 'running') LIMIT 1"
            ),
            {"type": JOB_TYPE, "run": run_id},
        ).scalar_one_or_none()
        is not None
    )
