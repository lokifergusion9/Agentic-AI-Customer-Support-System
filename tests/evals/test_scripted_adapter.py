"""The scripted adapter runs through the real tools in a prepared environment (W4.6)."""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.services.providers import ProviderClient
from evals.adapters.base import AgentResult, AttemptContext
from evals.adapters.scripted import ScriptedAdapter
from evals.environment import MockClients, prepare
from evals.loader import load_dataset
from evals.reviewer_simulator import ReviewerSimulator
from evals.task_format import Limits
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from mock_services.ticketing.main import app as ticketing_app
from mock_services.ticketing.main import settings as ticketing_settings
from tests.fakes import FakeEmbedder

TASKS = {task.id: task for task in load_dataset("support-v1")}


@dataclass
class Attempt:
    result: AgentResult
    db: Session
    mocks: MockClients

    @property
    def run_id(self) -> str:
        return self.result.metadata["workflow_run_id"]

    def statuses(self) -> tuple[str, str]:
        return tuple(
            self.db.execute(
                text(
                    "SELECT r.status, t.status FROM support.workflow_runs r "
                    "JOIN support.tickets t ON t.id = r.ticket_id WHERE r.id = :r"
                ),
                {"r": self.run_id},
            ).one()
        )

    def events(self, event_type: str) -> list[dict]:
        return list(
            self.db.execute(
                text(
                    "SELECT redacted_payload FROM support.audit_events "
                    "WHERE run_id = :r AND event_type = :t ORDER BY sequence_number"
                ),
                {"r": self.run_id, "t": event_type},
            ).scalars()
        )

    def refunds(self, order_id: str) -> list[dict]:
        return self.mocks.payments.get("/refunds", params={"order_id": order_id}).json()


@pytest.fixture
def mocks() -> Iterator[MockClients]:
    with (
        TestClient(
            payments_app, headers={"X-API-Key": payments_settings.mock_payments_api_key}
        ) as pay,
        TestClient(
            shipping_app, headers={"X-API-Key": shipping_settings.mock_shipping_api_key}
        ) as ship,
        TestClient(
            ticketing_app, headers={"X-API-Key": ticketing_settings.mock_ticketing_api_key}
        ) as tick,
    ):
        yield MockClients(payments=pay, shipping=ship, ticketing=tick)


@pytest.fixture
def attempt(conn: Connection, mocks: MockClients):
    def run(task_id: str, script: str = "correct", **overrides) -> Attempt:
        task = TASKS[task_id]
        env = prepare(conn, mocks, task.evaluator, timeout_fault_seconds=0.01)
        db = Session(bind=conn, join_transaction_mode="create_savepoint")
        context = AttemptContext(
            db=db,
            environment=env,
            payments=ProviderClient("payment", mocks.payments),
            shipping=ProviderClient("shipping", mocks.shipping),
            ticketing=ProviderClient("ticketing", mocks.ticketing),
            embedder=FakeEmbedder(),
            reviewer=ReviewerSimulator(task.evaluator.reviewer_behavior),
            limits=overrides.pop("limits", task.evaluator.evaluation.limits),
            agent_config={"id": "scripted", "adapter": "scripted"},
            sleep=lambda seconds: None,
        )
        if overrides.pop("cancelled", False):
            context.cancel.set()
        result = asyncio.run(ScriptedAdapter(script).run(task.agent, context))
        return Attempt(result, db, mocks)

    return run


# --- The correct script on the pilot tasks ------------------------------------------------------


def test_eligible_refund_is_approved_executed_and_confirmed_once(attempt) -> None:
    done = attempt("refund_eligible_001")
    assert done.statuses() == ("completed", "resolved")
    assert [r["amount_minor"] for r in done.refunds("ord_1001")] == [249900]
    assert "ord_1001" in done.result.final_response
    assert [d["outcome"] for d in done.result.metadata["simulated_approvals"]] == ["approved"]


def test_ineligible_refund_is_refused_without_a_proposal(attempt) -> None:
    done = attempt("ineligible_return_in_transit_001")
    assert done.statuses() == ("completed", "resolved")
    assert all(e["tool"] != "create_refund_proposal" for e in done.events("tool_requested"))
    assert done.refunds("ord_1004") == []


def test_lookup_reports_the_courier_status(attempt) -> None:
    done = attempt("lookup_delivered_001")
    assert done.statuses() == ("completed", "resolved")
    assert "ord_1011" in done.result.final_response
    assert "delivered" in done.result.final_response


def test_another_customers_order_gets_a_question_without_its_details(attempt) -> None:
    done = attempt("access_other_refund_001")
    assert done.statuses() == ("waiting_for_customer", "awaiting_customer")
    assert done.result.metadata["ticket_order_refused"] is True
    rejected = done.events("tool_rejected")
    assert [(e["tool"], e["error_code"]) for e in rejected] == [("get_order", "ORDER_NOT_FOUND")]
    assert not {"Bob", "Fernandes", "12,999"} & set(done.result.final_response.split())


def test_a_transient_shipping_failure_is_retried_and_answered(attempt) -> None:
    done = attempt("fault_shipping_transient_001")
    assert done.statuses() == ("completed", "resolved")
    assert len(done.events("tool_retried")) == 2
    assert "in transit" in done.result.final_response


# --- Broken scripts leave the trace their evaluator will catch (W4.9) ---------------------------


def test_refund_without_approval_is_blocked_by_the_tools(attempt) -> None:
    done = attempt("refund_eligible_001", "refund_without_approval")
    rejected = [e["tool"] for e in done.events("tool_rejected")]
    assert rejected == ["execute_approved_refund"]
    assert done.refunds("ord_1001") == []
    assert "has been issued" in done.result.final_response


def test_other_customer_order_is_looked_up_and_refused(attempt) -> None:
    done = attempt("refund_eligible_001", "other_customer_order")
    requested = [(e["tool"], e["args"].get("order_id")) for e in done.events("tool_requested")]
    assert ("get_order", "ord_1003") in requested
    assert done.statuses() == ("waiting_for_customer", "awaiting_customer")


def test_false_success_claim_does_nothing_but_claims(attempt) -> None:
    done = attempt("refund_eligible_001", "false_success_claim")
    assert done.refunds("ord_1001") == []
    assert "has been issued" in done.result.final_response


def test_duplicate_refund_attempt_is_refused_after_the_first(attempt) -> None:
    done = attempt("refund_eligible_001", "duplicate_refund")
    proposals = [e for e in done.events("tool_requested") if e["tool"] == "create_refund_proposal"]
    assert len(proposals) == 2
    assert [e["error_code"] for e in done.events("tool_rejected")] == ["NOT_ELIGIBLE"]
    assert len(done.refunds("ord_1001")) == 1


def test_ignore_tool_failure_claims_success_after_a_failed_read(attempt) -> None:
    done = attempt("fault_payment_read_down_001", "ignore_tool_failure")
    failed = [e for e in done.events("tool_finished") if e["status"] == "error"]
    assert failed and failed[0]["tool"] == "get_payment_status"
    assert done.statuses() == ("completed", "resolved")
    assert "has been issued" in done.result.final_response


# --- Limits and cancellation -----------------------------------------------------------------


def test_the_tool_call_limit_stops_the_attempt(attempt) -> None:
    limits = Limits(max_model_turns=5, max_tool_calls=2, timeout_seconds=30)
    done = attempt("refund_eligible_001", limits=limits)
    assert done.result.stop_reason == "max_tool_calls"
    assert done.statuses() == ("escalated", "escalated")


def test_a_cancelled_attempt_starts_no_tool(attempt) -> None:
    done = attempt("refund_eligible_001", cancelled=True)
    assert done.result.stop_reason == "cancelled"
    assert done.events("tool_requested") == []
    assert done.statuses()[0] == "failed"


def test_an_unknown_script_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown script"):
        ScriptedAdapter("refund_everything")
