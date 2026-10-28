"""Reconcile and resume after a restart (W3.12, HLD §5a).

The order matters: refund operations in an uncertain state are reconciled with the provider
**first**, so no run continues, retries or tells the customer anything on stale local state.
Then overdue approvals are expired, and unfinished runs continue from their checkpoints.
Every step is safe to repeat, so the sweep can run on every start or by hand with
`uv run python -m scripts.recover`.

The worker (W3.13) runs it with `queue_work=True`: instead of driving runs itself, it queues
one job per run (deduplicated by run), and leaves alone any run a job is already handling, so
two processes never drive the same run.
"""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SCHEMA
from app.services.approvals import expire_overdue_approvals
from app.services.audit import record_event
from app.services.refunds import confirm_refund
from app.workflows.dispatch import enqueue_decision, enqueue_drive, run_has_active_job
from app.workflows.runner import WorkflowServices, drive_run, resume_after_decision

ACTOR = "system:recovery"
UNCERTAIN = ("reserved", "unknown", "submitted")
QUEUED = "queued"


@dataclass
class RecoveryReport:
    # operation_id -> status after reconciling
    operations: dict[str, str] = field(default_factory=dict)
    expired_approvals: list[str] = field(default_factory=list)
    # run_id -> run status after resuming, or "queued" when a job will drive it
    runs: dict[str, str] = field(default_factory=dict)


def recover(
    db: Session,
    services: WorkflowServices,
    *,
    now: datetime | None = None,
    queue_work: bool = False,
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
        if queue_work and run_has_active_job(db, run_id):
            continue  # its job confirms it
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
        if queue_work:
            run_id = enqueue_decision(db, approval_id)
            db.commit()
            report.runs[run_id] = QUEUED
        else:
            run = resume_after_decision(db, approval_id, services)
            report.runs[run.id] = run.status

    # 3. Runs this process should be driving: continue each from its checkpoint.
    for (run_id,) in db.execute(
        text(f"SELECT id FROM {SCHEMA}.workflow_runs WHERE status = 'running' ORDER BY created_at")
    ).all():
        if run_id in report.runs:
            continue
        if queue_work:
            if not run_has_active_job(db, run_id):
                enqueue_drive(db, run_id)
                db.commit()
                report.runs[run_id] = QUEUED
        else:
            report.runs[run_id] = drive_run(db, run_id, services).status
    return report
