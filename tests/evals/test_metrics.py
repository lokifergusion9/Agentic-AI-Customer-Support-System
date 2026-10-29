"""Metrics and their denominators (W5.1, E§20)."""

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from evals import metrics
from evals.configs import scripted_config_row
from evals.execution import EvalContext
from evals.experiments import ExperimentPlan, create_experiment
from evals.loader import load_dataset
from evals.metrics import RunData, card, ratio, scorecards
from tests.evals.test_evaluators import Trace
from tests.evals.test_experiment_execution import run_all
from tests.fakes import FakeEmbedder

TASKS = {task.id: task.evaluator.model_dump(mode="json") for task in load_dataset("support-v1")}
T0 = datetime(2026, 10, 29, 10, 0, tzinfo=UTC)


def score(evaluator, dimension, status="pass", reason="OK", applicable=True, **evidence):
    return {
        "evaluator_name": evaluator,
        "dimension": dimension,
        "status": status,
        "reason_code": reason,
        "applicable": applicable,
        "evidence": evidence,
    }


def make_run(
    n: int = 1,
    *,
    task_id: str = "refund_eligible_001",
    config_id: str = "cfg",
    repetition: int = 1,
    success: bool | None = True,  # None: no finished attempt
    evaluation: str | None = None,
    execution: str = "completed",
    scores: list[dict[str, Any]] | None = None,
    trace: Trace | None = None,
    usage: dict[str, Any] | None = None,
    active_ms: int = 1000,
    wait_ms: int = 0,
    queue_s: float = 1.0,
    model_id: str = "model-a",
) -> RunData:
    task = TASKS[task_id]
    run = RunData(
        run_id=f"r{n}",
        config_id=config_id,
        model_id=model_id,
        task_id=task_id,
        task_version_id=f"{task_id}@v1",
        category=task["category"],
        repetition=repetition,
        execution_status=execution,
        evaluation_status=evaluation
        or ("pass" if success else "fail" if success is False else "not_applicable"),
        created_at=T0,
        task=task,
    )
    if success is not None:
        run.first_started_at = T0 + timedelta(seconds=queue_s)
        run.final = {
            "id": f"a{n}",
            "verified_success": success,
            "active_ms": active_ms,
            "approval_wait_ms": wait_ms,
            "finished_at": T0 + timedelta(seconds=30),
            "usage": usage
            if usage is not None
            else {"input_tokens": 1000, "output_tokens": 200, "cached_input_tokens": 0},
        }
    run.scores = scores or []
    run.events = (trace or Trace()).events
    return run


# --- Denominators ------------------------------------------------------------------------------


def test_success_coverage_and_strict_success_use_different_denominators() -> None:
    runs = [
        make_run(1, success=True),
        make_run(2, success=True),
        make_run(3, success=False),
        make_run(4, success=None, execution="infrastructure_error"),  # never scored
    ]
    result = card(runs)

    assert result["verified_success_rate"] == {"value": 2 / 3, "n": 2, "of": 3}
    assert result["strict_success_rate"] == {"value": 0.5, "n": 2, "of": 4}
    assert result["evaluation_coverage"] == {"value": 0.75, "n": 3, "of": 4}
    assert result["run_outcomes"]["infrastructure_error/not_applicable"] == 1


def test_high_success_with_low_coverage_is_not_equivalent_to_full_coverage() -> None:
    thin = [make_run(1, success=True)] + [
        make_run(i, success=None, execution="infrastructure_error") for i in range(2, 11)
    ]
    full = [make_run(i, success=True) for i in range(1, 11)]
    a, b = card(thin), card(full)
    assert a["verified_success_rate"]["value"] == b["verified_success_rate"]["value"] == 1.0
    assert a["strict_success_rate"]["value"] == 0.1 and b["strict_success_rate"]["value"] == 1.0
    assert a["evaluation_coverage"]["value"] == 0.1


def test_a_rate_over_nothing_is_unknown_not_zero() -> None:
    assert ratio(0, 0) == {"value": None, "n": 0, "of": 0}
    result = card([make_run(1, success=None, execution="cancelled")])
    assert result["verified_success_rate"]["value"] is None
    assert result["tokens_and_cost"]["tokens_per_successful_run"] is None


def test_an_evaluation_error_run_is_not_scorable() -> None:
    result = card([make_run(1, success=None, evaluation="evaluation_error")])
    assert result["scorable_runs"] == 0


# --- Safety ------------------------------------------------------------------------------------


def test_unsafe_attempts_and_executed_violations_are_counted_apart() -> None:
    blocked = [
        score("policy", "refund_requires_valid_approval", "fail", "X_ATTEMPT", severity="attempt")
    ]
    executed = [score("policy", "cross_customer_access", "fail", "X", severity="hard")]
    runs = [
        make_run(1, success=True, scores=[score("policy", "a")]),
        make_run(2, success=True, scores=blocked),  # tried, was blocked: still a success
        make_run(3, success=False, scores=executed),
        make_run(4, success=False, scores=blocked + executed),
    ]
    result = card(runs)
    assert result["unsafe_attempt_rate"] == {"value": 0.5, "n": 2, "of": 4}
    assert result["executed_violation_rate"] == {"value": 0.5, "n": 2, "of": 4}


def test_duplicate_refund_count_sums_the_extra_financial_effects() -> None:
    duplicate = [
        score(
            "outcome",
            "financial_effects",
            "fail",
            "DUPLICATE_FINANCIAL_EFFECT",
            expected_count=1,
            observed_count=3,
        )
    ]
    result = card([make_run(1, success=False, scores=duplicate), make_run(2, success=True)])
    assert result["duplicate_refund_count"] == 2


# --- Tools -------------------------------------------------------------------------------------


def test_tool_schema_validity_precision_and_recall() -> None:
    trace = (
        Trace()
        .call("get_order", order_id="ord_1001")
        .call("get_order", order_id="ord_1001")  # an exact repeat is not a correct new call
        .call("get_return_status", "rejected", "INVALID_ARGUMENTS", order="ord_1001")
        .call("get_shipment_status", order_id="ord_1001")  # not in the task's allowed actions
    )
    required = score(
        "trajectory",
        "required_steps",
        "partial",
        "MISSING_REQUIRED_STEP",
        required=["a", "b", "c", "d"],
        missing=["c"],
        recall=0.75,
    )
    result = card([make_run(1, success=True, trace=trace, scores=[required])])

    assert result["tool_schema_validity"] == {"value": 0.75, "n": 3, "of": 4}
    assert result["tool_call_precision"] == {"value": 0.25, "n": 1, "of": 4}
    assert result["required_call_recall"] == {"value": 0.75, "n": 3, "of": 4}


def test_workflow_calls_are_not_the_models_calls() -> None:
    trace = Trace()
    trace.add("tool_requested", "system:workflow", tool="update_ticket", call_id="c", args={})
    trace.add("tool_finished", "system:workflow", tool="update_ticket", call_id="c", status="ok")
    assert card([make_run(1, trace=trace)])["tool_schema_validity"]["of"] == 0


def test_policy_retrieval_counts_runs_that_found_the_required_section() -> None:
    def search(*citations: str) -> Trace:
        trace = Trace()
        trace.add("tool_requested", tool="search_policy", call_id="s", args={})
        trace.add(
            "tool_finished",
            tool="search_policy",
            call_id="s",
            status="ok",
            citations=list(citations),
        )
        return trace

    # refund_eligible_001 relies on refund_policy_v1 sections 3, 4 and 5.
    runs = [
        make_run(
            1,
            trace=search(
                "refund_policy_v1:section_3",
                "refund_policy_v1:section_4",
                "refund_policy_v1:section_5",
            ),
        ),
        make_run(2, trace=search("refund_policy_v1:section_3")),
        make_run(3, trace=search("shipping_delays_v1:section_4")),
        make_run(4),  # never searched
        make_run(5, task_id="missing_order_status_001"),  # relies on no policy: not counted
    ]
    result = card(runs)["policy_retrieval"]
    assert result["any_required_section"] == {"value": 0.5, "n": 2, "of": 4}
    assert result["all_required_sections"] == {"value": 0.25, "n": 1, "of": 4}


# --- Latency and resolution kinds --------------------------------------------------------------


def test_queue_active_and_reviewer_wait_are_separate_distributions() -> None:
    runs = [
        make_run(i, queue_s=q, active_ms=a, wait_ms=w)
        for i, (q, a, w) in enumerate([(1, 1000, 10), (2, 2000, 20), (3, 3000, 30)], start=1)
    ]
    latency = card(runs)["latency"]
    assert latency["queue_ms"]["p50"] == 2000
    assert latency["active_ms"] == {"n": 3, "p50": 2000, "p95": 2900, "max": 3000}
    assert latency["approval_wait_ms"]["max"] == 30


def test_human_assisted_resolution_is_reported_apart_from_automated() -> None:
    assisted = Trace().proposal()
    runs = [make_run(1, trace=assisted), make_run(2, success=False, trace=assisted), make_run(3)]
    split = card(runs)["resolution"]
    assert split["human_assisted"]["runs"] == 2 and split["human_assisted"]["successful"] == 1
    assert split["fully_automated"] == {
        "runs": 1,
        "successful": 1,
        "success_rate": {"value": 1.0, "n": 1, "of": 1},
    }


# --- Tokens and cost ---------------------------------------------------------------------------


PRICE = {
    "provider": "gemini",
    "model_id": "model-a",
    "effective_from": date(2026, 1, 1),
    "input_per_mtok_usd": 0.10,
    "output_per_mtok_usd": 0.40,
    "cached_input_per_mtok_usd": None,
}


def test_tokens_and_estimated_cost_per_successful_run() -> None:
    runs = [make_run(1, success=True), make_run(2, success=False)]
    usage = card(runs, [PRICE])["tokens_and_cost"]
    assert usage["total_tokens"] == 2400
    assert usage["tokens_per_successful_run"] == 2400  # all candidate tokens over successes
    assert usage["estimated_cost_per_successful_run_usd"] == pytest.approx(
        2 * (0.1 * 1000 + 0.4 * 200) / 1e6
    )
    assert "estimate" in usage["cost_label"]


def test_missing_usage_is_unknown_cost_never_free() -> None:
    runs = [make_run(1, usage={"input_tokens": None, "output_tokens": None}), make_run(2)]
    usage = card(runs, [PRICE])["tokens_and_cost"]
    assert usage["runs_with_unknown_usage"] == 1 and usage["runs_with_unknown_cost"] == 1
    assert usage["tokens_per_successful_run"] is None
    assert usage["estimated_cost_per_successful_run_usd"] is None


def test_without_a_registered_price_the_cost_is_unknown() -> None:
    usage = card([make_run(1)], [])["tokens_and_cost"]
    assert usage["runs_with_unknown_cost"] == 1
    assert usage["estimated_cost_per_successful_run_usd"] is None
    assert usage["tokens_per_successful_run"] == 1200  # tokens are still known


def test_the_price_in_effect_on_the_run_date_is_used() -> None:
    later = {**PRICE, "effective_from": date(2027, 1, 1), "input_per_mtok_usd": 99}
    assert metrics.price_for([PRICE, later], "model-a", date(2026, 10, 29)) is PRICE
    assert metrics.price_for([PRICE], "other-model", None) is None


# --- Repetitions -------------------------------------------------------------------------------


def repeated(results: dict[str, list[bool]]) -> list[RunData]:
    runs, n = [], 0
    for task_id, outcomes in results.items():
        for repetition, ok in enumerate(outcomes, start=1):
            n += 1
            runs.append(make_run(n, task_id=task_id, repetition=repetition, success=ok))
    return runs


def test_repetitions_report_the_task_mean_flakiness_and_passed_all() -> None:
    result = card(
        repeated(
            {
                "refund_eligible_001": [True, True, True],
                "lookup_delivered_001": [True, False, True],
                "access_other_refund_001": [False, False, False],
            }
        )
    )["repetitions"]

    assert result["tasks"] == 3
    assert result["mean_task_success"] == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert result["passed_all_repetitions"] == {"value": 1 / 3, "n": 1, "of": 3}
    assert result["flaky_tasks"] == ["lookup_delivered_001"]
    assert result["per_task"]["lookup_delivered_001"] == {"runs": 3, "successes": 2, "mean": 2 / 3}


def test_the_bootstrap_resamples_tasks_and_is_reproducible() -> None:
    means = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5]
    first, second = metrics.bootstrap_ci(means), metrics.bootstrap_ci(means)
    assert first == second
    low, high = first
    assert 0 <= low < sum(means) / len(means) < high <= 1
    assert metrics.bootstrap_ci([1.0]) is None  # one task: no uncertainty to estimate


def test_repetitions_of_one_task_do_not_narrow_the_interval() -> None:
    once = card(repeated({"refund_eligible_001": [True], "lookup_delivered_001": [False]}))
    thrice = card(
        repeated({"refund_eligible_001": [True] * 3, "lookup_delivered_001": [False] * 3})
    )
    assert once["repetitions"]["ci95"] == thrice["repetitions"]["ci95"]


# --- Per configuration and category, from the database -----------------------------------------


def test_scorecards_group_by_configuration_and_category() -> None:
    runs = [
        make_run(1, config_id="a", success=True),
        make_run(2, config_id="a", task_id="lookup_delivered_001", success=False),
        make_run(3, config_id="b", success=True),
    ]
    cards = scorecards(runs)
    assert set(cards) == {"a", "b"}
    assert cards["a"]["overall"]["planned_runs"] == 2
    assert set(cards["a"]["by_category"]) == {"eligible_refund", "read_only_lookup"}
    assert cards["a"]["by_category"]["read_only_lookup"]["verified_success_rate"]["value"] == 0.0


@pytest.fixture
def session(conn: Connection) -> Session:
    return Session(bind=conn, join_transaction_mode="create_savepoint")


def test_a_scripted_experiment_produces_the_scorecard(session, mocks) -> None:
    ctx = EvalContext(
        mocks=mocks, embedder=FakeEmbedder(), timeout_fault_seconds=0.01, sleep=lambda s: None
    )
    plan = ExperimentPlan(
        "m",
        "support-v1",
        "pilot",
        [scripted_config_row("correct"), scripted_config_row("false_success_claim")],
        1,
        "metrics",
    )
    experiment_id, _ = create_experiment(session, plan)
    run_all(session, ctx)

    cards = metrics.experiment_scorecard(session.connection(), experiment_id)

    good, bad = (
        cards["scripted_correct"]["overall"],
        cards["scripted_false_success_claim"]["overall"],
    )
    assert good["verified_success_rate"] == {"value": 1.0, "n": 5, "of": 5}
    assert good["evaluation_coverage"]["value"] == 1.0
    assert good["tool_schema_validity"]["value"] == 1.0
    assert good["resolution"]["human_assisted"]["runs"] == 1  # only the refund asked for approval
    assert good["tokens_and_cost"]["total_tokens"] == 0  # a scripted agent calls no model
    assert bad["verified_success_rate"]["value"] < 1.0
    assert "response.refund_claim:FALSE_REFUND_CLAIM" in bad["failure_reasons"]
    assert set(cards["scripted_correct"]["by_category"]) == {
        "eligible_refund",
        "ineligible_refund",
        "read_only_lookup",
        "access_boundary",
        "tool_failure",
    }


def test_only_the_final_attempt_of_a_run_counts(session, mocks) -> None:
    ctx = EvalContext(
        mocks=mocks, embedder=FakeEmbedder(), timeout_fault_seconds=0.01, sleep=lambda s: None
    )
    experiment_id, _ = create_experiment(
        session,
        ExperimentPlan("m", "support-v1", "pilot", [scripted_config_row("correct")], 1, "final"),
    )
    run_all(session, ctx)
    session.execute(  # an earlier, interrupted attempt of one run
        text(
            "INSERT INTO eval.attempts (id, run_id, attempt_number, status, environment_id, "
            "finished_at) SELECT 'att_old', min(id), 99, 'interrupted', 'env_old', now() "
            "FROM eval.runs"
        )
    )
    runs = metrics.load_runs(session.connection(), experiment_id)
    assert all(r.final["status"] == "finished" for r in runs)
    assert len(runs) == 5 and sum(r.success for r in runs) == 5


def test_registered_prices_feed_the_cost_estimate(session) -> None:
    from evals.prices import register_price

    connection = session.connection()
    register_price(
        connection,
        provider="gemini",
        model_id="model-a",
        effective_from=date(2026, 1, 1),
        input_per_mtok_usd=0.1,
        output_per_mtok_usd=0.4,
        source="test fixture",
    )
    prices = metrics.load_prices(connection)
    assert [p["model_id"] for p in prices] == ["model-a"]
    assert card([make_run(1)], prices)["tokens_and_cost"]["runs_with_unknown_cost"] == 0
    with pytest.raises(ValueError, match="where the rate was read"):
        register_price(
            connection,
            provider="gemini",
            model_id="m",
            effective_from=date(2026, 1, 1),
            input_per_mtok_usd=1,
            output_per_mtok_usd=1,
            source=" ",
        )
