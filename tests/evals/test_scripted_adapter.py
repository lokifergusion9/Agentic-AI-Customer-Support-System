"""The scripted adapter runs through the real tools in a prepared environment (W4.6)."""

import pytest

from evals.adapters.scripted import ScriptedAdapter
from evals.task_format import Limits


@pytest.fixture
def attempt(run_attempt):
    def run(task_id: str, script: str = "correct", **options):
        return run_attempt(task_id, ScriptedAdapter(script), **options)

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
