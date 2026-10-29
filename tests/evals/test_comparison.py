"""Comparing two configurations on the same tasks (W5.3)."""

import pytest
from sqlalchemy import Connection
from sqlalchemy.orm import Session

from evals import experiments
from evals.comparison import IncompatibleExperiments, Side, compare, sign_test_p, task_change
from evals.configs import scripted_config_row
from evals.execution import EvalContext
from evals.experiments import ExperimentPlan, create_experiment
from tests.evals.test_experiment_execution import run_all
from tests.evals.test_metrics import make_run
from tests.fakes import FakeEmbedder


@pytest.fixture
def session(conn: Connection) -> Session:
    return Session(bind=conn, join_transaction_mode="create_savepoint")


@pytest.fixture
def ctx(mocks) -> EvalContext:
    return EvalContext(
        mocks=mocks, embedder=FakeEmbedder(), timeout_fault_seconds=0.01, sleep=lambda s: None
    )


# --- One task ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "change"),
    [
        ([True], [False], "regressed"),
        ([False], [True], "improved"),
        ([True], [True], "both_pass"),
        ([False], [False], "both_fail"),
        ([True, False, True], [True, True, True], "improved"),  # a flaky task that got steadier
        ([True, True, True], [True, False, True], "regressed"),
        ([True, False], [False, True], "tied"),
    ],
)
def test_a_task_is_compared_on_its_mean_success(a, b, change) -> None:
    runs_a = [make_run(i, repetition=i, success=ok) for i, ok in enumerate(a, start=1)]
    runs_b = [make_run(10 + i, repetition=i, success=ok) for i, ok in enumerate(b, start=1)]
    assert task_change("refund_eligible_001", runs_a, runs_b)["change"] == change


def test_a_task_one_side_could_not_score_is_not_comparable() -> None:
    unscored = [make_run(1, success=None, execution="infrastructure_error")]
    result = task_change("refund_eligible_001", unscored, [make_run(2, success=True)])
    assert result["change"] == "not_comparable"
    assert result["baseline"]["scorable_runs"] == 0


def test_a_regression_carries_the_failures_that_explain_it() -> None:
    failing = make_run(
        2,
        success=False,
        scores=[
            {
                "evaluator_name": "response",
                "dimension": "refund_claim",
                "status": "fail",
                "reason_code": "FALSE_REFUND_CLAIM",
                "applicable": True,
                "evidence": {},
            }
        ],
    )
    result = task_change("refund_eligible_001", [make_run(1)], [failing])
    assert result["change"] == "regressed"
    assert result["candidate"]["failures"] == ["response.refund_claim:FALSE_REFUND_CLAIM"]
    assert result["candidate"]["run_ids"] == ["r2"]


def test_the_sign_test() -> None:
    assert sign_test_p(0, 0) is None
    assert sign_test_p(5, 0) == pytest.approx(0.0625)
    assert sign_test_p(3, 3) == 1.0
    assert sign_test_p(1, 0) == 1.0  # one differing task proves nothing


# --- Two configurations ------------------------------------------------------------------------


def two_configs(session, ctx, key="cmp", split="pilot", scripts=("correct", "false_success_claim")):
    plan = ExperimentPlan(
        "c", "support-v1", split, [scripted_config_row(s) for s in scripts], 1, key
    )
    experiment_id, _ = create_experiment(session, plan)
    run_all(session, ctx)
    return experiment_id


def test_the_comparison_names_the_tasks_that_regressed_and_improved(session, ctx) -> None:
    experiment_id = two_configs(session, ctx)
    good, bad = "scripted_correct", "scripted_false_success_claim"
    connection = session.connection()

    worse = compare(connection, Side(experiment_id, good), Side(experiment_id, bad))
    assert worse["compatible"] and worse["tasks_compared"] == 5
    # This script claims "refund issued" on every refund message, so it fails the ineligible
    # refund too, not only the eligible one.
    assert worse["regressed_tasks"] == ["ineligible_return_in_transit_001", "refund_eligible_001"]
    assert worse["improved_tasks"] == []
    changed = next(t for t in worse["tasks"] if t["task_id"] == "refund_eligible_001")
    assert "response.refund_claim:FALSE_REFUND_CLAIM" in changed["candidate"]["failures"]
    assert worse["summary"]["regressed"] == 2 and worse["summary"]["sign_test_p"] == 0.5

    better = compare(connection, Side(experiment_id, bad), Side(experiment_id, good))
    assert (
        better["improved_tasks"] == ["ineligible_return_in_transit_001", "refund_eligible_001"]
        and better["regressed_tasks"] == []
    )


def test_the_differences_cover_success_coverage_tools_tokens_and_latency(session, ctx) -> None:
    experiment_id = two_configs(session, ctx)
    result = compare(
        session.connection(),
        Side(experiment_id, "scripted_correct"),
        Side(experiment_id, "scripted_false_success_claim"),
    )
    differences = result["differences"]
    assert set(differences) >= {
        "verified_success_rate",
        "strict_success_rate",
        "evaluation_coverage",
        "executed_violation_rate",
        "tool_calls_per_run",
        "model_calls_per_run",
        "tokens_per_run",
        "estimated_cost_per_successful_run_usd",
        "active_ms_p50",
        "active_ms_p95",
    }
    success = differences["verified_success_rate"]
    assert success["baseline"] == 1.0 and success["candidate"] == pytest.approx(0.6)
    assert success["difference"] == pytest.approx(-0.4)
    assert differences["estimated_cost_per_successful_run_usd"]["difference"] is None  # no price
    assert (
        differences["tool_calls_per_run"]["candidate"]
        < differences["tool_calls_per_run"]["baseline"]
    )
    assert result["by_category"]["eligible_refund"]["verified_success_rate"]["difference"] == -1.0


def test_one_configuration_per_experiment_needs_no_name(session, ctx) -> None:
    a = two_configs(session, ctx, key="a", scripts=("correct",))
    b = two_configs(session, ctx, key="b", scripts=("false_success_claim",))
    result = compare(session.connection(), Side(a), Side(b))
    assert result["regressed_tasks"] == ["ineligible_return_in_transit_001", "refund_eligible_001"]
    assert result["baseline"]["experiment_id"] == a and result["candidate"]["experiment_id"] == b


def test_an_experiment_with_several_configurations_needs_one_named(session, ctx) -> None:
    experiment_id = two_configs(session, ctx)
    with pytest.raises(ValueError, match="name the one to compare"):
        compare(session.connection(), Side(experiment_id), Side(experiment_id, "scripted_correct"))
    with pytest.raises(ValueError, match="no configuration"):
        compare(
            session.connection(),
            Side(experiment_id, "nope"),
            Side(experiment_id, "scripted_correct"),
        )


# --- Compatibility -----------------------------------------------------------------------------


def test_experiments_on_different_splits_are_not_compared(session, ctx) -> None:
    pilot = two_configs(session, ctx, key="p", scripts=("correct",))
    plan = ExperimentPlan("v", "support-v1", "validation", [scripted_config_row("correct")], 1, "v")
    validation, _ = create_experiment(session, plan)
    with pytest.raises(IncompatibleExperiments) as caught:
        compare(session.connection(), Side(pilot), Side(validation))
    assert any(d.startswith("split") for d in caught.value.differences)


def test_experiments_with_different_evaluator_versions_are_not_compared(
    session, ctx, monkeypatch
) -> None:
    first = two_configs(session, ctx, key="e1", scripts=("correct",))
    versions = [(name, "9.9.9", fn) for name, _, fn in experiments.EVALUATORS]
    monkeypatch.setattr(experiments, "EVALUATORS", versions)
    plan = ExperimentPlan(
        "e", "support-v1", "pilot", [scripted_config_row("false_success_claim")], 1, "e2"
    )
    second, _ = create_experiment(session, plan)
    with pytest.raises(IncompatibleExperiments, match="evaluator_versions"):
        compare(session.connection(), Side(first), Side(second))
