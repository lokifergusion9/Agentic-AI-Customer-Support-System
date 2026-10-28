"""The audit trail as a replayable trace (W3.15, E§19)."""

import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from app.db import SCHEMA
from app.services.audit import record_event
from app.services.runs import list_run_events
from app.workflows.runner import start_resolution
from scripts.seed_data import seed
from tests.integration import test_workflow as workflow
from tests.integration.test_proposals import ALICE, TODAY
from tests.integration.test_workflow import (
    APPROVED_IN_CHAT,
    HEADPHONES,
    SPECIALIST,
    headphones_scenario,
    open_ticket,
    refund_ids,
)

# The workflow tests' fixtures: seeded database, mock providers, scripted services.
db = workflow.db
providers = workflow.providers
make_services = workflow.make_services

# Words that would mean private model reasoning was stored (only token counts may be).
REASONING_KEYS = {"thought", "thoughts", "reasoning", "chain_of_thought", "thinking"}


def keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys(v)}
    return set()


def run_events(db: Session, services, message: str = HEADPHONES):
    run = start_resolution(db, ALICE, open_ticket(db, message, "ord_1001"), services)
    return run, list_run_events(db, run.id)


def test_events_are_numbered_in_order_without_gaps(db: Session, make_services, providers) -> None:
    payments, _ = providers
    _, events = run_events(db, make_services(headphones_scenario(*refund_ids(db, payments))))

    numbers = [e["sequence_number"] for e in events]
    assert numbers == list(range(1, len(events) + 1))
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)


def test_the_turn_ends_with_agent_finished(db: Session, make_services, providers) -> None:
    payments, _ = providers
    _, events = run_events(db, make_services(headphones_scenario(*refund_ids(db, payments))))

    finished = [e for e in events if e["event_type"] == "agent_finished"]
    assert len(finished) == 1 and events[-1]["event_type"] == "agent_finished"
    payload = finished[0]["payload"]
    assert (payload["route"], payload["stop_reason"]) == ("request_approval", "model_final")
    assert "No refund has been issued yet" in payload["final_response"]


def test_a_budget_stop_is_the_stop_reason(db: Session, make_services, providers) -> None:
    payments, _ = providers
    services = make_services(
        headphones_scenario(*refund_ids(db, payments)), replace(SPECIALIST, max_steps=3)
    )
    _, events = run_events(db, services)

    [finished] = [e["payload"] for e in events if e["event_type"] == "agent_finished"]
    assert (finished["stop_reason"], finished["halted_by"]) == ("max_steps", "max_steps")


def test_a_blocked_attempt_is_distinguishable_from_an_executed_effect(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    scenario = headphones_scenario(*refund_ids(db, payments))
    scenario.plans["investigation"] = [
        [("execute_approved_refund", {"proposal_id": "prp_anything"})],
        *scenario.plans["investigation"],
    ]
    _, events = run_events(db, make_services(scenario), APPROVED_IN_CHAT)

    rejected = [e["payload"] for e in events if e["event_type"] == "tool_rejected"]
    finished = [e["payload"] for e in events if e["event_type"] == "tool_finished"]
    assert [r["tool"] for r in rejected] == ["execute_approved_refund"]
    assert rejected[0]["error_code"]
    assert "execute_approved_refund" not in {f["tool"] for f in finished}
    assert all(isinstance(f["duration_ms"], int) for f in finished)
    requested = [e["payload"]["call_id"] for e in events if e["event_type"] == "tool_requested"]
    assert len(requested) == len(rejected) + len(finished)  # every attempt has one outcome


def test_no_private_model_reasoning_is_stored(db: Session, make_services, providers) -> None:
    payments, _ = providers
    _, events = run_events(db, make_services(headphones_scenario(*refund_ids(db, payments))))

    assert not keys([e["payload"] for e in events]) & REASONING_KEYS
    assert all(e["created_at"].utcoffset() == timedelta(0) for e in events)  # stored in UTC


@pytest.fixture
def committed(engine: Engine) -> Iterator[Engine]:
    with engine.begin() as connection:
        seed(connection, TODAY)
    yield engine
    with engine.begin() as cleanup:
        cleanup.execute(
            text(
                f"TRUNCATE {SCHEMA}.customers, {SCHEMA}.policy_documents, "
                f"{SCHEMA}.audit_events CASCADE"
            )
        )


def test_concurrent_writers_for_one_run_get_distinct_numbers(committed: Engine) -> None:
    """Separate connections (the API and the worker, say) appending to the same run."""
    start = threading.Barrier(4)

    def writer(name: str) -> None:
        start.wait()
        for n in range(10):
            with Session(committed) as session:
                record_event(
                    session, actor=name, event_type="probe", payload={"n": n}, run_id="run_1002"
                )
                session.commit()

    threads = [threading.Thread(target=writer, args=(f"system:w{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    with committed.connect() as connection:
        numbers = list(
            connection.execute(
                text(
                    f"SELECT sequence_number FROM {SCHEMA}.audit_events "
                    "WHERE run_id = 'run_1002' ORDER BY sequence_number"
                )
            ).scalars()
        )
    assert numbers == list(range(1, len(numbers) + 1))
    assert len(numbers) >= 40
