"""Reconcile and resume after a restart (W3.12, HLD §5a).

The order matters: refund operations in an uncertain state are reconciled with the provider
**first**, so no run continues, retries or tells the customer anything on stale local state.
Then overdue approvals are expired, and unfinished runs continue from their checkpoints.
Every step is safe to repeat, so the sweep can run on every start (W3.13 runs it before the
worker takes jobs) or by hand with `uv run python -m scripts.recover`.
"""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SCHEMA
from app.services.approvals import expire_overdue_approvals
from app.services.audit import record_event
from app.services.refunds import confirm_refund
from app.services.runs import get_run
from app.workflows.runner import (
    WorkflowServices,
    invoke,
    paused_state,
    restore_budget,
    resume_after_decision,
    resume_after_provider,
    run_customer,
    runtime,
)

ACTOR = "system:recovery"
UNCERTAIN = ("reserved", "unknown", "submitted")
# Waits for a person, not for this process: these runs are left alone.
EXTERNAL_WAITS = {"await_customer", "await_approval"}


@dataclass
class RecoveryReport:
    # operation_id -> status after reconciling
    operations: dict[str, str] = field(default_factory=dict)
    expired_approvals: list[str] = field(default_factory=list)
    # run_id -> run status after resuming
    runs: dict[str, str] = field(default_factory=dict)


def recover(
    db: Session, services: WorkflowServices, *, now: datetime | None = None
) -> RecoveryReport:
    report = RecoveryReport()

    # 1. Money first: bring every uncertain refund in line with the provider.
    for operation_id, proposal_id, run_id in db.execute(
        text(
            f"SELECT o.id, o.proposal_id, p.run_id FROM {SCHEMA}.refund_operations o "
            f"JOIN {SCHEMA}.action_proposals p ON p.id = o.proposal_id "
            "WHERE o.status = ANY(:uncertain) ORDER BY o.created_at"
        ),
        {"uncertain": list(UNCERTAIN)},
    ).all():
        operation = confirm_refund(db, services.payments, proposal_id)
        report.operations[operation_id] = operation.status
        record_event(
            db,
            actor=ACTOR,
            event_type="refund_recovered",
            payload={"operation_id": operation_id, "status": operation.status},
            run_id=run_id,
        )
        db.commit()

    # 2. Approvals nobody decided in time: the run renews them or escalates (W3.6).
    report.expired_approvals = expire_overdue_approvals(db, now=now)
    for approval_id in report.expired_approvals:
        run = resume_after_decision(db, approval_id, services)
        report.runs[run.id] = run.status

    # 3. Runs this process should be driving: continue each from its checkpoint.
    for (run_id,) in db.execute(
        text(f"SELECT id FROM {SCHEMA}.workflow_runs WHERE status = 'running' ORDER BY created_at")
    ).all():
        if run_id not in report.runs:
            report.runs[run_id] = continue_run(db, run_id, services)
    return report


def continue_run(db: Session, run_id: str, services: WorkflowServices) -> str:
    """Resume one `running` run from its checkpoint; returns its status afterwards."""
    customer = run_customer(db, run_id)
    run = get_run(db, customer, run_id)
    rt = runtime(db, customer, run, services, steps_used=0, tokens_used=0)
    waiting_at = set(paused_state(rt, services, run).next)
    if "await_provider" in waiting_at:
        return resume_after_provider(db, run_id, services).status
    if not waiting_at or waiting_at & EXTERNAL_WAITS:
        return run.status
    # Stopped between nodes (a crash): every node is idempotent, so run on from there.
    rt.recorder.record(ACTOR, "run_resumed", {"after": "restart", "next": sorted(waiting_at)})
    restore_budget(rt, services, run)
    return invoke(db, customer, run, rt, services, None).status
