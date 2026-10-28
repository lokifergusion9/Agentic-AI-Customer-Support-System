"""Start and resume resolution runs (W2.13, W2.14, W3.3, W3.13).

Each entry point is split in two: a short step that records the request (`begin_resolution`,
`accept_message`; the reviewer's decision is saved by the approval service) and a step that
drives the graph (`drive_run`, `continue_with_message`, `resume_after_decision`). Inline
mode runs both in the request; worker mode runs the second on the worker (see
`app/workflows/dispatch.py`). The driving steps are safe to repeat: each checks where the run's
checkpoint stands, so a retried or recovered job never repeats finished work.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, StateSnapshot
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.agents.config import AgentConfig
from app.agents.loop import AgentRuntime
from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.llm.calls import RunBudget, RunRecorder
from app.llm.types import ModelProvider
from app.policies.embeddings import Embedder
from app.services.approvals import get_approval
from app.services.audit import record_event
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient
from app.services.runs import RunView, get_run, set_run_output, set_run_status, start_run
from app.services.tickets import get_ticket
from app.tools.base import ToolContext
from app.workflows.graph import build_graph

logger = logging.getLogger(__name__)

# Pauses that wait for a person, not for this process: driving a run leaves them alone.
EXTERNAL_WAITS = {"await_customer", "await_approval"}
TERMINAL = {"completed", "escalated", "failed"}

PROMPTS = {
    "single": ["classifier", "single_agent"],
    "specialist": ["classifier", "investigation", "policy", "resolution"],
}


@dataclass
class WorkflowServices:
    """What a run needs from outside; the API builds it from settings, tests from fakes."""

    provider: ModelProvider
    config: AgentConfig
    payments: ProviderClient
    shipping: ProviderClient
    embedder: Embedder
    checkpointer: BaseCheckpointSaver
    today: date | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)


def start_resolution(
    db: Session, customer: AuthenticatedCustomer, ticket_id: str, services: WorkflowServices
) -> RunView:
    run = begin_resolution(db, customer, ticket_id, services.config)
    return drive_run(db, run.id, services)


def begin_resolution(
    db: Session, customer: AuthenticatedCustomer, ticket_id: str, config: AgentConfig
) -> RunView:
    """Create the run (status `running`) and record its configuration; the graph starts later."""
    get_ticket(db, customer, ticket_id)  # TICKET_NOT_FOUND unless it is theirs
    run = start_run(db, customer, ticket_id)  # RUN_ALREADY_ACTIVE if one is unfinished
    RunRecorder(db, run.id, run.ticket_id).record(
        "system:workflow", "run_configured", config.summary(PROMPTS[config.architecture])
    )
    return run


def drive_run(db: Session, run_id: str, services: WorkflowServices) -> RunView:
    """Run the graph as far as it goes from wherever the run's checkpoint stands.

    A run with no checkpoint starts from the ticket. A run waiting for the payment provider
    asks it again. A run stopped between nodes (a crash, a lost worker) continues from there;
    every node is idempotent. A run waiting for a person, or finished, is left as it is.
    """
    customer = run_customer(db, run_id)
    run = get_run(db, customer, run_id)
    if run.status in TERMINAL:
        return run
    rt = runtime(db, customer, run, services, steps_used=0, tokens_used=0)
    snapshot = paused_state(rt, services, run)
    if not snapshot.values:
        ticket = get_ticket(db, customer, run.ticket_id)
        initial: dict[str, Any] = {
            "run_id": run.id,
            "ticket_id": ticket.id,
            "authenticated_customer_id": customer.customer_id,
            "order_id": ticket.order_id,
            "customer_message": ticket.customer_message,
            "customer_messages": [ticket.customer_message],
            "clarification_rounds": 0,
            "verified_facts": {},
            "policy_references": [],
            "retry_counts": {},
        }
        return invoke(db, customer, run, rt, services, initial)
    waiting_at = set(snapshot.next)
    if "await_provider" in waiting_at:
        return resume_after_provider(db, run_id, services)
    approval_id = snapshot.values.get("approval_id")
    if (
        "await_approval" in waiting_at
        and approval_id
        and get_approval(db, approval_id).decision is not None
    ):
        # Decided, but the resume was lost (e.g. the decision job failed): resume it now.
        return resume_after_decision(db, approval_id, services)
    if not waiting_at or waiting_at & EXTERNAL_WAITS:
        return run
    rt.recorder.record(
        "system:workflow", "run_resumed", {"after": "interruption", "next": sorted(waiting_at)}
    )
    restore_budget(rt, services, run)
    return invoke(db, customer, run, rt, services, None)


def resume_with_message(
    db: Session,
    customer: AuthenticatedCustomer,
    run_id: str,
    message: str,
    services: WorkflowServices,
) -> RunView:
    """W2.13: the customer answers a clarification question; the same thread continues."""
    accept_message(db, customer, run_id, message)
    return continue_with_message(db, run_id, message, services)


def accept_message(
    db: Session, customer: AuthenticatedCustomer, run_id: str, message: str
) -> RunView:
    """Record the customer's reply and mark the run running; the graph resumes later."""
    run = get_run(db, customer, run_id)  # RUN_NOT_FOUND unless it is theirs
    if run.status != "waiting_for_customer":
        raise ServiceFailure("RUN_NOT_WAITING", "This run is not waiting for a customer reply.")
    actor = f"customer:{customer.customer_id}"
    set_run_status(db, run_id, "running", actor=actor)
    # Redacted: the reply text goes to the workflow state, only its length to the audit trail.
    record_event(
        db,
        actor=actor,
        event_type="customer_message",
        payload={"chars": len(message)},
        ticket_id=run.ticket_id,
        run_id=run_id,
    )
    db.commit()
    return get_run(db, customer, run_id)


def continue_with_message(
    db: Session, run_id: str, message: str, services: WorkflowServices
) -> RunView:
    """Resume the clarification wait with the reply; if it was already resumed, drive on."""
    customer = run_customer(db, run_id)
    run = get_run(db, customer, run_id)
    rt = runtime(db, customer, run, services, steps_used=0, tokens_used=0)
    if "await_customer" not in paused_state(rt, services, run).next:
        return drive_run(db, run_id, services)
    restore_budget(rt, services, run)
    return invoke(db, customer, run, rt, services, Command(resume=message))


def resume_after_decision(db: Session, approval_id: str, services: WorkflowServices) -> RunView:
    """W3.3: wake the run paused on this approval, on the same thread, once it is decided.

    The caller (the reviewer's decision endpoint, W3.4) has already stored the decision. The
    workflow reads that stored decision itself, so the resume value authorizes nothing. Resuming
    a run that is no longer paused on the approval changes nothing, so a repeated or late
    resume can never execute the refund twice.
    """
    owner = db.execute(
        text(
            "SELECT a.decision, p.run_id, c.id AS customer_id, c.external_auth_subject "
            f"FROM {SCHEMA}.approvals a "
            f"JOIN {SCHEMA}.action_proposals p ON p.id = a.proposal_id "
            f"JOIN {SCHEMA}.customers c ON c.id = p.customer_id WHERE a.id = :approval"
        ),
        {"approval": approval_id},
    ).one_or_none()
    if owner is None:
        raise ServiceFailure("APPROVAL_NOT_FOUND", "No approval with this ID was found.")
    if owner.decision is None:
        raise ServiceFailure("APPROVAL_PENDING", "This approval has not been decided yet.")
    # The run belongs to the customer; tools run on their behalf, as before the pause.
    customer = AuthenticatedCustomer(owner.customer_id, owner.external_auth_subject)
    run = get_run(db, customer, owner.run_id)
    rt = runtime(db, customer, run, services, steps_used=0, tokens_used=0)
    snapshot = paused_state(rt, services, run)
    if "await_approval" not in snapshot.next or snapshot.values.get("approval_id") != approval_id:
        return run
    rt.recorder.record(
        "system:workflow",
        "run_resumed",
        {"approval_id": approval_id, "decision": owner.decision},
    )
    restore_budget(rt, services, run)
    return invoke(db, customer, run, rt, services, Command(resume={"approval_id": approval_id}))


def resume_after_provider(db: Session, run_id: str, services: WorkflowServices) -> RunView:
    """W3.11/W3.12: check a run waiting for the provider to confirm its refund again.

    The workflow asks the provider itself; the resume value carries nothing. A run that is
    not paused on the provider is returned unchanged.
    """
    customer = run_customer(db, run_id)
    run = get_run(db, customer, run_id)
    rt = runtime(db, customer, run, services, steps_used=0, tokens_used=0)
    if "await_provider" not in paused_state(rt, services, run).next:
        return run
    rt.recorder.record("system:workflow", "run_resumed", {"waiting_for": "provider"})
    restore_budget(rt, services, run)
    return invoke(db, customer, run, rt, services, Command(resume={"check": "provider"}))


def run_customer(db: Session, run_id: str) -> AuthenticatedCustomer:
    """The customer a run acts for: tools run on their behalf, whoever resumes it."""
    owner = db.execute(
        text(
            "SELECT c.id, c.external_auth_subject "
            f"FROM {SCHEMA}.workflow_runs r JOIN {SCHEMA}.tickets t ON t.id = r.ticket_id "
            f"JOIN {SCHEMA}.customers c ON c.id = t.customer_id WHERE r.id = :run"
        ),
        {"run": run_id},
    ).one_or_none()
    if owner is None:
        raise ServiceFailure("RUN_NOT_FOUND", "No run with this ID was found.")
    return AuthenticatedCustomer(owner.id, owner.external_auth_subject)


def paused_state(rt: AgentRuntime, services: WorkflowServices, run: RunView) -> StateSnapshot:
    graph = build_graph(rt).compile(checkpointer=services.checkpointer)
    return graph.get_state({"configurable": {"thread_id": run.thread_id}})


def restore_budget(rt: AgentRuntime, services: WorkflowServices, run: RunView) -> None:
    # The budget is per run, not per request: continue from what the checkpoint recorded.
    values = paused_state(rt, services, run).values or {}
    rt.budget.steps_used = values.get("steps_used", 0)
    rt.budget.tokens_used = values.get("tokens_used", 0)


def runtime(
    db: Session,
    customer: AuthenticatedCustomer,
    run: RunView,
    services: WorkflowServices,
    *,
    steps_used: int,
    tokens_used: int,
) -> AgentRuntime:
    recorder = RunRecorder(db, run.id, run.ticket_id)
    ctx = ToolContext(
        db=db,
        customer=customer,
        run_id=run.id,
        ticket_id=run.ticket_id,
        payments=services.payments,
        shipping=services.shipping,
        embedder=services.embedder,
        today=services.today,
        lock=recorder.lock,
    )
    budget = RunBudget(
        services.config.max_steps, services.config.max_tokens, steps_used, tokens_used
    )
    return AgentRuntime(services.provider, services.config, recorder, budget, ctx, services.sleep)


def invoke(
    db: Session,
    customer: AuthenticatedCustomer,
    run: RunView,
    rt: AgentRuntime,
    services: WorkflowServices,
    graph_input: Any,
) -> RunView:
    graph = build_graph(rt).compile(checkpointer=services.checkpointer)
    try:
        graph.invoke(graph_input, {"configurable": {"thread_id": run.thread_id}})
    except Exception as exc:
        # Unexpected failure (not a model limit, which the graph escalates itself): the run is
        # marked failed with the error class, never left "running".
        logger.exception("run %s failed", run.id)
        db.rollback()
        set_run_status(db, run.id, "failed", actor="system:workflow")
        set_run_output(
            db, run.id, {"route": "failed", "stop_reason": "error", "error": type(exc).__name__}
        )
        db.commit()
    return get_run(db, customer, run.id)
