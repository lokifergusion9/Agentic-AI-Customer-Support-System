"""Reconcile and resume after a restart (W3.12, HLD §5a)."""

from datetime import UTC, datetime, timedelta

from langgraph.types import Command
from sqlalchemy.orm import Session

from app.services.approvals import decide_approval
from app.services.runs import get_run
from app.services.tickets import get_ticket
from app.workflows.graph import build_graph
from app.workflows.recovery import recover
from app.workflows.runner import runtime
from tests.integration import test_workflow as workflow
from tests.integration.test_approvals import RIYA
from tests.integration.test_proposals import ALICE
from tests.integration.test_workflow import (
    awaiting_review,
    event_types,
    paused_at,
    pending_refund,
    settle,
)

# The workflow tests' fixtures: seeded database, mock providers, scripted services.
db = workflow.db
providers = workflow.providers
make_services = workflow.make_services

# Seeded: ord_1003's refund, which the payment mock keeps pending (W1.2).
SEEDED_PENDING = "rop_1003"


def crash_after_the_provider_call(db: Session, run, services) -> None:
    """Approve and resume, but stop the process right after execute_refund (a crash)."""
    decide_approval(db, RIYA, run.output["approval_id"], "approved")
    rt = runtime(db, ALICE, run, services, steps_used=0, tokens_used=0)
    graph = build_graph(rt).compile(
        checkpointer=services.checkpointer, interrupt_before=["verify_refund"]
    )
    graph.invoke(
        Command(resume={"approval_id": run.output["approval_id"]}),
        {"configurable": {"thread_id": run.thread_id}},
    )


def test_uncertain_refund_is_reconciled_before_the_run_continues(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    run, services = awaiting_review(db, make_services, providers)
    payments.http.post("/admin/faults", json={"mode": "error"})  # 503: the outcome is unknown
    crash_after_the_provider_call(db, run, services)
    assert paused_at(db, run, services) == ("verify_refund",)
    assert payments.get("/payments/ord_1001")["refunds"] == []

    report = recover(db, services)

    assert report.runs == {run.id: "completed"}
    operation_id = get_run(db, ALICE, run.id).output["refund_operation_id"]
    assert report.operations[operation_id] == "succeeded"
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1
    events = event_types(db, run.id)
    # The money is settled before the workflow runs on and tells the customer.
    assert events.index("refund_recovered") < events.index("run_resumed")
    assert get_ticket(db, ALICE, run.ticket_id).status == "resolved"


def test_refund_the_provider_already_has_is_not_sent_again(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    run, services = awaiting_review(db, make_services, providers)
    crash_after_the_provider_call(db, run, services)  # the provider succeeded
    payments.http.post("/admin/faults", json={"mode": "error", "retryable": False})

    report = recover(db, services)

    assert report.runs == {run.id: "completed"}
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1


def test_run_waiting_for_the_provider_finishes_once_it_settles(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    waiting, services = pending_refund(db, make_services, providers)
    assert recover(db, services).runs == {waiting.id: "running"}  # still pending

    settle(payments, "succeeded")
    report = recover(db, services)

    assert report.runs == {waiting.id: "completed"}
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1


def test_overdue_approval_nobody_decided_is_expired_and_renewed(
    db: Session, make_services, providers
) -> None:
    run, services = awaiting_review(db, make_services, providers)

    report = recover(db, services, now=datetime.now(UTC) + timedelta(hours=2))

    assert report.expired_approvals == [run.output["approval_id"]]
    assert report.runs == {run.id: "waiting_for_approval"}
    assert "approval_renewed" in event_types(db, run.id)


def test_recovery_is_safe_to_repeat_and_leaves_waiting_runs_alone(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    run, services = awaiting_review(db, make_services, providers)
    crash_after_the_provider_call(db, run, services)
    recover(db, services)

    again = recover(db, services)

    # Only the seeded refund the provider still holds as pending is checked again.
    assert again.operations == {SEEDED_PENDING: "submitted"}
    assert (again.expired_approvals, again.runs) == ([], {})
    assert len(payments.get("/payments/ord_1001")["refunds"]) == 1
