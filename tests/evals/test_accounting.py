"""Usage accounting, and the default run making no model calls (W4.10, E§28)."""

import tomllib
from pathlib import Path

from evals.adapters.base import AgentResult
from evals.execution import execution_status, usage


def call(outcome="ok", input_tokens=100, output_tokens=20, cached=0, retry=0):
    return {
        "event_type": "model_call",
        "payload": {
            "outcome": outcome,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached,
            "retry": retry,
        },
    }


def test_retried_calls_contribute_their_usage() -> None:
    # A request that was retried after a 429: the failed one reports no usage, the retry does.
    trace = [call("error", None, None, None), call(retry=1), call()]
    totals = usage(trace)
    assert totals["model_calls"] == 3
    assert totals["retried_calls"] == 1
    assert (totals["input_tokens"], totals["output_tokens"]) == (200, 40)


def test_missing_usage_stays_unknown_and_is_never_zero() -> None:
    totals = usage([call(), call(input_tokens=None)])
    assert totals["input_tokens"] is None  # one call did not report: the total is unknown
    assert totals["output_tokens"] == 40
    assert usage([])["input_tokens"] == 0  # no calls at all is a known zero


def test_non_model_events_are_ignored() -> None:
    trace = [{"event_type": "tool_finished", "payload": {}}, call()]
    assert usage(trace)["model_calls"] == 1


def test_a_turn_limit_stop_is_a_completed_attempt_with_its_reason() -> None:
    result = AgentResult("Your request needs a specialist.", "max_steps", {})
    assert execution_status(result, timed_out=False) == "completed"


def test_budget_timeout_and_cancellation_have_their_own_statuses() -> None:
    assert execution_status(AgentResult("", "token_limit", {}), False) == "budget_exceeded"
    assert execution_status(AgentResult("", "model_final", {}), True) == "timed_out"
    assert execution_status(AgentResult("", "cancelled", {}), False) == "cancelled"


def test_the_default_test_run_excludes_model_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    options = config["tool"]["pytest"]["ini_options"]
    assert "not llm" in options["addopts"]
    assert any(marker.startswith("llm:") for marker in options["markers"])
