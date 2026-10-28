"""Job handlers, looked up by `job_type` (W3.13). W4.8 registers `eval_attempt` here."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from app.jobs.queue import Job
from app.services.helpdesk import retry_sync
from app.workflows.runner import WorkflowServices, continue_with_message, drive_run


@dataclass
class HandlerContext:
    services: WorkflowServices
    engine: Engine


Handler = Callable[[Session, Job, HandlerContext], None]
HANDLERS: dict[str, Handler] = {}


def register(job_type: str) -> Callable[[Handler], Handler]:
    def add(handler: Handler) -> Handler:
        HANDLERS[job_type] = handler
        return handler

    return add


@contextmanager
def run_lock(engine: Engine, run_id: str) -> Iterator[None]:
    """One job at a time per run, across workers; released if the worker dies.

    Each job re-reads the run's checkpoint once it holds the lock, so jobs for the same run
    (a message and a sweep, say) run one after the other and never drive the run twice.
    """
    with engine.connect() as connection:
        key = {"key": f"run:{run_id}"}
        connection.execute(text("SELECT pg_advisory_lock(hashtextextended(:key, 0))"), key)
        connection.commit()
        try:
            yield
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(hashtextextended(:key, 0))"), key)
            connection.commit()


@register("workflow_run")
def handle_workflow_run(db: Session, job: Job, ctx: HandlerContext) -> None:
    payload = job.payload
    run_id = payload["run_id"]
    with run_lock(ctx.engine, run_id):
        if payload["action"] == "message":
            continue_with_message(db, run_id, payload["message"], ctx.services)
        else:
            # drive (start, recovery) and decision: drive_run resumes a decided approval too.
            drive_run(db, run_id, ctx.services)


@register("ticket_sync")
def handle_ticket_sync(db: Session, job: Job, ctx: HandlerContext) -> None:
    """Retry only the ticket update the helpdesk missed (W3.14); raises while it is down."""
    if ctx.services.ticketing is None:
        raise RuntimeError("no ticketing client configured")
    retry_sync(db, ctx.services.ticketing, job.payload)
