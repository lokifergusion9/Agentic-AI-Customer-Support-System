"""The deterministic reviewer simulator decides through the real approval service (W4.5)."""

from itertools import count

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.approvals import decide_approval
from app.services.errors import ServiceFailure
from app.workflows.runner import resume_after_decision
from evals.reviewer_simulator import SIMULATOR_REVIEWERS, ReviewerSimulator
from evals.task_format import ReviewerBehavior
from tests.integration import test_workflow as workflow
from tests.integration.test_proposals import BOB, NOW, propose
from tests.integration.test_workflow import awaiting_review

db = workflow.db
providers = workflow.providers
make_services = workflow.make_services


def behavior(decision: str, order_id: str = "ord_1001", amount: int = 249900) -> ReviewerBehavior:
    return ReviewerBehavior.model_validate(
        {
            "decision": decision,
            "applies_to": {"action": "refund", "order_id": order_id, "amount_minor": amount},
        }
    )


def refunds(providers) -> list[dict]:
    payments, _ = providers
    return payments.get("/payments/ord_1001")["refunds"]


def decided_by(db: Session, approval_id: str) -> tuple[str, str, str]:
    row = db.execute(
        text("SELECT decision, reviewer_id, reviewer_role FROM support.approvals WHERE id = :id"),
        {"id": approval_id},
    ).one()
    return tuple(row)


def test_a_matching_proposal_is_approved_and_refunded_once(db, make_services, providers) -> None:
    run, services = awaiting_review(db, make_services, providers)
    simulator = ReviewerSimulator(behavior("approve"))

    approval_id = simulator.pending_approval(db, run.id)
    decision = simulator.decide(db, approval_id)
    done = resume_after_decision(db, approval_id, services)

    assert (decision.outcome, decision.matched, decision.behavior) == ("approved", True, "approve")
    assert decided_by(db, approval_id) == ("approved", "sim|reviewer", "reviewer")
    assert (done.status, done.output["route"]) == ("completed", "refunded")
    assert len(refunds(providers)) == 1
    assert simulator.pending_approval(db, run.id) is None


@pytest.mark.parametrize(
    "permitted",
    [
        behavior("approve", amount=100000),  # a different amount
        behavior("approve", order_id="ord_1002"),  # a different order
    ],
)
def test_anything_but_the_permitted_action_is_rejected(
    db, make_services, providers, permitted
) -> None:
    run, services = awaiting_review(db, make_services, providers)
    simulator = ReviewerSimulator(permitted)

    decision = simulator.decide(db, run.output["approval_id"])
    done = resume_after_decision(db, run.output["approval_id"], services)

    assert (decision.outcome, decision.matched) == ("rejected", False)
    assert (done.status, done.output["route"]) == ("completed", "rejected")
    assert refunds(providers) == []


def test_reject_rejects_even_the_permitted_action(db, make_services, providers) -> None:
    run, _ = awaiting_review(db, make_services, providers)
    decision = ReviewerSimulator(behavior("reject")).decide(db, run.output["approval_id"])
    assert (decision.outcome, decision.matched) == ("rejected", True)


def test_a_task_without_reviewer_behavior_rejects_any_approval(db, make_services, providers):
    run, _ = awaiting_review(db, make_services, providers)
    decision = ReviewerSimulator(None).decide(db, run.output["approval_id"])
    assert (decision.behavior, decision.outcome) == ("none", "rejected")


def test_expire_lets_every_approval_lapse_until_the_run_escalates(
    db, make_services, providers
) -> None:
    run, services = awaiting_review(db, make_services, providers)
    simulator = ReviewerSimulator(behavior("expire"))

    # What the adapter will do (W4.7): decide each wait, resume, until the run stops waiting.
    while (approval_id := simulator.pending_approval(db, run.id)) is not None:
        simulator.decide(db, approval_id)
        run = resume_after_decision(db, approval_id, services)

    assert [d.outcome for d in simulator.decisions] == ["expired"] * 3  # MAX_APPROVAL_ROUNDS
    assert all(d.simulated_delay_seconds > 0 for d in simulator.decisions)
    assert decided_by(db, simulator.decisions[0].approval_id) == ("expired", None, None)
    assert run.status == "escalated"
    assert refunds(providers) == []


def test_a_senior_only_approval_is_decided_by_a_simulated_senior(db, providers) -> None:
    payments, _ = providers
    _, _, proposal = propose(db, payments, BOB, "ord_1014")
    simulator = ReviewerSimulator(behavior("approve", order_id="ord_1014", amount=8999900))

    decision = simulator.decide(db, proposal.approval_id, now=NOW)

    assert (decision.required_role, decision.outcome) == ("senior_reviewer", "approved")
    assert decided_by(db, proposal.approval_id) == (
        "approved",
        "sim|senior_reviewer",
        "senior_reviewer",
    )


def test_the_service_still_checks_the_simulators_role(db, providers) -> None:
    payments, _ = providers
    _, _, proposal = propose(db, payments, BOB, "ord_1014")
    with pytest.raises(ServiceFailure, match="senior reviewer"):
        decide_approval(
            db, SIMULATOR_REVIEWERS["reviewer"], proposal.approval_id, "approved", now=NOW
        )


def test_waiting_time_is_measured_apart_from_execution(db, make_services, providers) -> None:
    run, _ = awaiting_review(db, make_services, providers)
    ticks = count(start=10.0, step=0.25)
    simulator = ReviewerSimulator(behavior("approve"), clock=lambda: next(ticks))

    decision = simulator.decide(db, run.output["approval_id"])

    assert decision.wait_ms == simulator.total_wait_ms == 250.0
    assert decision.as_event()["simulated"] is True
