"""One test per failure in PDF §14 (W3.16, HLD 5b): the failure, and the expected behavior.

Everything is real except the model, which is scripted: tools, services, mock providers,
the business-rule validator, the graph, its checkpoints and the recovery sweep.
"""

from dataclasses import replace

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.approvals import decide_approval
from app.services.tickets import get_ticket
from app.workflows.recovery import recover
from app.workflows.runner import resume_after_decision, start_resolution
from tests.integration import test_helpdesk as helpdesk
from tests.integration import test_workflow as workflow
from tests.integration.test_approvals import RIYA
from tests.integration.test_policy_search import REFUND_QUESTION
from tests.integration.test_proposals import ALICE
from tests.integration.test_recovery import crash_after_the_provider_call
from tests.integration.test_workflow import (
    HEADPHONES,
    SPECIALIST,
    Scenario,
    awaiting_review,
    event_types,
    expire,
    headphones_scenario,
    open_ticket,
    refund_ids,
)

# The workflow tests' fixtures: seeded database, mock providers, scripted services.
db = workflow.db
providers = workflow.providers
make_services = workflow.make_services
ticketing = helpdesk.ticketing


def headphones(db: Session, providers) -> Scenario:
    payments, _ = providers
    return headphones_scenario(*refund_ids(db, payments))


def refunds(providers) -> list[dict]:
    payments, _ = providers
    return payments.get("/payments/ord_1001")["refunds"]


def proposals(db: Session, run_id: str) -> int:
    return db.execute(
        text("SELECT count(*) FROM support.action_proposals WHERE run_id = :r"), {"r": run_id}
    ).scalar_one()


# --- Input and access ------------------------------------------------------------------------


def test_missing_order_information_asks_a_targeted_question(
    db: Session, make_services, providers
) -> None:
    base = headphones(db, providers)
    scenario = replace(
        base,
        outputs={**base.outputs, "classifier": {**base.outputs["classifier"], "order_id": None}},
        plans={**base.plans, "investigation": [[("list_orders", {})]]},
    )

    run = start_resolution(
        db, ALICE, open_ticket(db, "Where is my refund?", None), make_services(scenario)
    )

    assert run.status == "waiting_for_customer"
    question = run.output["final_response"]
    assert "order ID" in question and "ord_1001" in question  # names what is missing
    assert proposals(db, run.id) == 0


def test_order_of_another_customer_is_denied_without_details(db: Session, make_services) -> None:
    scenario = Scenario(
        outputs={
            "classifier": {
                "category": "refund_request",
                "order_id": "ord_1003",
                "summary": "Refund.",
            },
            "investigation": {
                "order_id": None,
                "summary": "Order not found.",
                "evidence_refs": [],
                "missing_information": ["order_id"],
            },
            "policy": {"interpretation": "", "policy_refs": [], "needs_human_review": False},
        },
        plans={"investigation": [[("get_order", {"order_id": "ord_1003"})]]},
    )

    run = start_resolution(
        db, ALICE, open_ticket(db, "Refund ord_1003", None), make_services(scenario)
    )

    rejected = db.execute(
        text(
            "SELECT redacted_payload FROM support.audit_events "
            "WHERE run_id = :r AND event_type = 'tool_rejected'"
        ),
        {"r": run.id},
    ).scalar_one()
    assert rejected["error_code"] == "ORDER_NOT_FOUND"  # the same answer as a missing order
    assert "ord_1003" not in run.output["final_response"]
    assert "Bob" not in str(run.output)


# --- Policy ----------------------------------------------------------------------------------


def test_no_applicable_policy_escalates_instead_of_inventing_eligibility(
    db: Session, make_services, providers
) -> None:
    scenario = headphones(db, providers)
    scenario.plans["policy"] = [
        [("search_policy", {"query": REFUND_QUESTION, "effective_on": "2024-06-01"})]
    ]
    scenario.outputs["policy"] = {
        "interpretation": "No refund policy was in effect.",
        "policy_refs": [],
        "needs_human_review": True,
    }
    # The model proposes anyway, citing a passage it never retrieved.

    run = start_resolution(
        db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), make_services(scenario)
    )

    assert run.status == "escalated"
    assert "UNKNOWN_POLICY_REF:refund_policy_v1:section_3" in run.output["validation_problems"]
    assert proposals(db, run.id) == 0 and refunds(providers) == []


def test_conflicting_policy_evidence_is_flagged_for_review(
    db: Session, make_services, providers
) -> None:
    scenario = headphones(db, providers)
    # One search mixes two versions of the refund policy; a later one is filtered correctly.
    scenario.plans["policy"] = [
        [("search_policy", {"query": REFUND_QUESTION, "policy_keys": ["refund_policy"]})],
        *scenario.plans["policy"],
    ]

    run = start_resolution(
        db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), make_services(scenario)
    )

    # The model did not flag it (needs_human_review is False); code did.
    assert run.status == "escalated"
    assert run.output["validation_problems"] == ["POLICY_CONFLICT"]
    assert proposals(db, run.id) == 0


# --- Model and tools -------------------------------------------------------------------------


def order_status() -> Scenario:
    return Scenario(
        outputs={
            "classifier": {"category": "order_status", "order_id": "ord_1008", "summary": "Where?"},
            "investigation": {
                "order_id": "ord_1008",
                "summary": "In transit.",
                "evidence_refs": ["ord_1008"],
                "missing_information": [],
            },
            "policy": {"interpretation": "", "policy_refs": [], "needs_human_review": False},
            "resolution": {
                "action": "answer_only",
                "order_id": "ord_1008",
                "reason": "Status request",
                "evidence_refs": ["ord_1008"],
                "customer_message": "Your order ord_1008 is on its way.",
            },
        },
        plans={
            "investigation": [
                [("get_order", {"order_id": "ord_1008"})],
                [("get_shipment_status", {"order_id": "ord_1008"})],
            ],
        },
    )


def test_temporary_read_api_failure_is_retried_with_bounded_backoff(
    db: Session, make_services, providers
) -> None:
    _, shipping = providers
    shipping.http.post("/admin/faults", json={"mode": "error"})  # one 503 on the next read
    waits: list[float] = []
    services = replace(make_services(order_status()), sleep=waits.append)

    run = start_resolution(db, ALICE, open_ticket(db, "Where is my order?", "ord_1008"), services)

    assert run.status == "completed"
    assert waits == [0.5]
    assert event_types(db, run.id).count("tool_retried") == 1


def test_a_read_api_that_stays_down_stops_after_the_bounded_retries(
    db: Session, make_services, providers
) -> None:
    _, shipping = providers
    for _ in range(3):
        shipping.http.post("/admin/faults", json={"mode": "error"})
    waits: list[float] = []
    services = replace(make_services(order_status()), sleep=waits.append)

    run = start_resolution(db, ALICE, open_ticket(db, "Where is my order?", "ord_1008"), services)

    assert waits == [0.5, 1.0]  # bounded: two retries, then the error goes to the agent
    finished = db.execute(
        text(
            "SELECT redacted_payload FROM support.audit_events WHERE run_id = :r "
            "AND event_type = 'tool_finished' AND redacted_payload->>'tool' = 'get_shipment_status'"
        ),
        {"r": run.id},
    ).scalar_one()
    assert (finished["status"], finished["attempts"]) == ("error", 3)


def test_invalid_model_output_gets_limited_repair_then_escalates(
    db: Session, make_services, providers
) -> None:
    scenario = headphones(db, providers)
    scenario.outputs["resolution"] = "this is not json"

    run = start_resolution(
        db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), make_services(scenario)
    )

    assert (run.status, run.output["stop_reason"]) == ("escalated", "invalid_model_output")
    assert event_types(db, run.id).count("model_output_invalid") == 2  # first try + one repair


def test_excessive_workflow_loops_stop_at_the_step_limit(
    db: Session, make_services, providers
) -> None:
    services = make_services(headphones(db, providers), replace(SPECIALIST, max_steps=3))

    run = start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)

    assert (run.status, run.output["stop_reason"]) == ("escalated", "max_steps")
    assert refunds(providers) == []


# --- Approval --------------------------------------------------------------------------------


def test_rejected_approval_is_recorded_and_follows_the_rejection_path(
    db: Session, make_services, providers
) -> None:
    run, services = awaiting_review(db, make_services, providers)

    decide_approval(db, RIYA, run.output["approval_id"], "rejected")
    done = resume_after_decision(db, run.output["approval_id"], services)

    assert (done.status, done.output["route"]) == ("completed", "rejected")
    assert "approval_decided" in event_types(db, run.id)
    assert refunds(providers) == []


def test_expired_approval_requires_a_new_valid_approval(
    db: Session, make_services, providers
) -> None:
    run, services = awaiting_review(db, make_services, providers)
    expire(db, run.output["approval_id"])

    waiting = resume_after_decision(db, run.output["approval_id"], services)

    assert waiting.status == "waiting_for_approval"
    assert waiting.output["approval_id"] != run.output["approval_id"]
    assert refunds(providers) == []


# --- Execution -------------------------------------------------------------------------------


def test_uncertain_refund_outcome_is_reconciled_before_claiming_success(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    run, services = awaiting_review(db, make_services, providers)
    # A 503 on the refund request: did the provider make the refund or not? Unknown.
    # (The accept-then-timeout case, on a real socket, is in test_recovery / test_refund_execution.)
    payments.http.post("/admin/faults", json={"mode": "error"})
    decide_approval(db, RIYA, run.output["approval_id"], "approved")

    done = resume_after_decision(db, run.output["approval_id"], services)

    assert (done.status, done.output["route"]) == ("completed", "refunded")
    assert len(refunds(providers)) == 1
    outcomes = [
        row.redacted_payload["status"]
        for row in db.execute(
            text(
                "SELECT redacted_payload FROM support.audit_events WHERE run_id = :r "
                "AND event_type = 'refund_executed' ORDER BY sequence_number"
            ),
            {"r": run.id},
        )
    ]
    # Unknown first; the provider was asked by key (no record), the same key was sent again,
    # and only then was success told to the customer.
    assert outcomes == ["unknown", "succeeded"]
    events = event_types(db, run.id)
    told = len(events) - 1 - events[::-1].index("agent_finished")  # the final turn's end
    assert events.index("refund_executed") < told


def test_refund_succeeds_but_ticket_update_fails_retries_only_the_ticket(
    db: Session, make_services, providers, ticketing
) -> None:
    services = replace(make_services(headphones(db, providers)), ticketing=ticketing)
    run = start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)
    decide_approval(db, RIYA, run.output["approval_id"], "approved")
    ticketing.http.post("/admin/faults", json={"mode": "error"})
    resume_after_decision(db, run.output["approval_id"], services)
    assert get_ticket(db, ALICE, run.ticket_id).status == "awaiting_approval"

    report = recover(db, services)

    assert report.ticket_syncs == {run.ticket_id: "synced"}
    assert get_ticket(db, ALICE, run.ticket_id).status == "resolved"
    assert len(refunds(providers)) == 1


def test_worker_crash_resumes_from_persisted_state_and_reconciles(
    db: Session, make_services, providers
) -> None:
    run, services = awaiting_review(db, make_services, providers)
    crash_after_the_provider_call(db, run, services)  # killed before verifying the refund

    report = recover(db, services)  # what the restarted worker runs first

    assert report.runs == {run.id: "completed"}
    assert len(refunds(providers)) == 1
    assert get_ticket(db, ALICE, run.ticket_id).status == "resolved"
