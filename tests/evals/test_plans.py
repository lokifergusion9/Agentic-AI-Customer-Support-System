"""Configurations, plans, quota deferral and the quota estimate (W5.2)."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.jobs.handlers import HandlerContext
from app.jobs.worker import WorkerConfig, process_next
from app.llm.types import ModelError
from evals.adapters.langgraph import LangGraphAdapter
from evals.configs import REGISTRY, named_config
from evals.execution import EvalContext, install, quota_resume_at
from evals.experiments import ExperimentPlan, create_experiment
from evals.loader import split_ids
from evals.plans import estimate_quota, load_plan, model_calls_per_run, planned_runs, quota_hits
from tests.evals.test_experiment_execution import claim
from tests.fakes import FakeEmbedder, ScriptedModel

DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


@pytest.fixture
def session(conn: Connection) -> Session:
    return Session(bind=conn, join_transaction_mode="create_savepoint")


def differing(a: str, b: str) -> set[str]:
    left, right = named_config(a), named_config(b)
    return {k for k in left if k not in ("id", "config_hash") and left[k] != right[k]}


# --- Configurations ----------------------------------------------------------------------------


def test_the_four_registered_configurations() -> None:
    assert set(REGISTRY) == {
        "lite_single_v1",
        "lite_specialist_v1",
        "lite_specialist_v2",
        "flash_specialist_v1",
    }
    assert {named_config(name)["id"] for name in REGISTRY} == set(REGISTRY)


def test_each_comparison_changes_one_thing() -> None:
    # Architecture: the architecture, and the prompts that architecture reads.
    assert differing("lite_single_v1", "lite_specialist_v1") == {"architecture", "prompt_hashes"}
    # Prompt: the version, and the content hashes of the prompts that changed.
    assert differing("lite_specialist_v1", "lite_specialist_v2") == {
        "prompt_version",
        "prompt_hashes",
    }
    # Model: the model and nothing else.
    assert differing("lite_specialist_v1", "flash_specialist_v1") == {"model_id"}


def test_prompt_v2_changes_the_investigation_and_resolution_prompts_only() -> None:
    v1, v2 = (
        named_config("lite_specialist_v1")["prompt_hashes"],
        named_config("lite_specialist_v2")["prompt_hashes"],
    )
    assert {name for name in v1 if v1[name] != v2[name]} == {"investigation", "resolution"}
    single1, single2 = named_config("lite_single_v1"), named_config("lite_single_v1")
    assert single1["config_hash"] == single2["config_hash"]  # deterministic


# --- Plans -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plan", "runs", "configs"),
    [
        ("pilot", 5, ["lite_specialist_v1"]),
        ("main", 150, ["lite_single_v1", "lite_specialist_v1", "lite_specialist_v2"]),
        ("repetitions", 60, ["lite_single_v1", "lite_specialist_v1"]),
        ("model_comparison", 10, ["flash_specialist_v1"]),
    ],
)
def test_the_plans_match_the_experiment_plan(session, plan, runs, configs) -> None:
    loaded = load_plan(plan)
    assert planned_runs(session.connection(), loaded) == runs
    assert [row["id"] for row in loaded.configs] == configs


def test_all_is_every_task_of_the_dataset() -> None:
    assert len(split_ids("support-v1", "all")) == 50


def test_the_held_out_test_split_needs_an_explicit_acknowledgement(session) -> None:
    final = load_plan("final")
    with pytest.raises(ValueError, match="held out"):
        create_experiment(session, final)
    allowed = load_plan("final", allow_test_split=True)
    _, created = create_experiment(session, allowed)
    assert created


def test_every_configuration_runs_the_same_tasks_under_the_same_conditions(session) -> None:
    plan = ExperimentPlan(
        "same",
        "support-v1",
        "pilot",
        [
            named_config("lite_single_v1"),
            named_config("lite_specialist_v1"),
            named_config("lite_specialist_v2"),
        ],
        2,
        "same-conditions",
    )
    experiment_id, _ = create_experiment(session, plan)

    rows = session.execute(
        text(
            "SELECT agent_config_id, task_version_id, count(*) AS runs FROM eval.runs "
            "WHERE experiment_id = :id GROUP BY 1, 2"
        ),
        {"id": experiment_id},
    ).all()
    by_config: dict[str, set] = {}
    for row in rows:
        by_config.setdefault(row.agent_config_id, set()).add((row.task_version_id, row.runs))
    first = next(iter(by_config.values()))
    assert len(by_config) == 3 and all(tasks == first for tasks in by_config.values())
    # Limits, reviewer behaviour and faults live in the task version, shared by every config.
    versions = session.execute(
        text("SELECT count(DISTINCT task_version_id) FROM eval.runs WHERE experiment_id = :id"),
        {"id": experiment_id},
    ).scalar_one()
    assert versions == 5


# --- Quota -------------------------------------------------------------------------------------


def test_a_daily_quota_resumes_after_the_next_pacific_midnight() -> None:
    now = datetime(2026, 10, 29, 20, 0, tzinfo=UTC)  # 13:00 Pacific
    resume = quota_resume_at({"quota": DAILY}, now)
    local = resume.astimezone(ZoneInfo("America/Los_Angeles"))
    assert (local.day, local.hour, local.minute) == (30, 0, 5)
    assert resume > now


def test_a_per_minute_quota_waits_what_the_provider_asked_but_at_least_a_minute() -> None:
    now = datetime(2026, 10, 29, 20, 0, tzinfo=UTC)
    assert quota_resume_at({"quota": "PerMinute", "retry_after_s": 12}, now) == now + timedelta(
        seconds=60
    )
    assert quota_resume_at({"quota": None, "retry_after_s": 300}, now) == now + timedelta(
        seconds=300
    )


def exhausted_quota(config) -> ScriptedModel:
    error = ModelError("HTTP_429", "quota", retryable=True, detail=DAILY)
    return ScriptedModel(lambda call: error, model_id=config.model_id)


@pytest.fixture
def quota_ctx(mocks) -> EvalContext:
    return EvalContext(
        mocks=mocks,
        embedder=FakeEmbedder(),
        adapter_for=lambda row: LangGraphAdapter(exhausted_quota),
        timeout_fault_seconds=0.01,
        sleep=lambda seconds: None,
    )


def test_an_exhausted_daily_quota_defers_the_job_and_loses_no_run(session, conn, quota_ctx) -> None:
    create_experiment(
        session,
        ExperimentPlan(
            "q", "support-v1", "pilot", [named_config("lite_specialist_v1")], 1, "quota"
        ),
    )
    install(quota_ctx)
    ctx = HandlerContext(None, None)  # type: ignore[arg-type]
    config = WorkerConfig(worker_id="w", lease_seconds=600, job_types=["eval_attempt"])

    assert process_next(
        lambda: Session(bind=conn, join_transaction_mode="create_savepoint"), ctx, config
    )

    # The attempt is an infrastructure error with its trace kept; the run waits, not fails.
    attempt = session.execute(
        text("SELECT status, execution_status, error FROM eval.attempts")
    ).one()
    assert (attempt.status, attempt.execution_status) == ("error", "infrastructure_error")
    assert attempt.error == {"provider_error": "HTTP_429", "quota": DAILY}
    statuses = session.execute(
        text("SELECT execution_status, evaluation_status FROM eval.runs ORDER BY id")
    ).all()
    assert {tuple(s) for s in statuses} == {("queued", "pending")}
    assert session.execute(text("SELECT status FROM eval.experiments")).scalar_one() == "running"

    # No retry was used up, and every queued run of this model is held back to the same time.
    jobs = session.execute(
        text("SELECT status, attempts, available_at, last_error FROM support.jobs")
    ).all()
    assert all(j.status == "queued" for j in jobs)
    assert min(j.attempts for j in jobs) == 0
    assert len({j.available_at for j in jobs}) == 1
    assert jobs[0].available_at > datetime.now(UTC) + timedelta(minutes=30)
    assert "quota exhausted" in next(j.last_error for j in jobs if j.last_error)
    assert claim(session) is None  # nothing can start before the quota is back

    # The 429s are recorded, so the first hit of the daily limit can be read from the traces.
    hits = quota_hits(conn)
    assert hits and hits[0]["quota"] == DAILY


# --- Estimating the quota ----------------------------------------------------------------------


def test_the_quota_estimate_is_pilot_calls_per_run_times_planned_runs(session, mocks) -> None:
    pilot_id, _ = create_experiment(
        session,
        ExperimentPlan("p", "support-v1", "pilot", [named_config("scripted_correct")], 1, "p"),
    )
    from tests.evals.test_experiment_execution import run_all

    run_all(
        session,
        EvalContext(
            mocks=mocks, embedder=FakeEmbedder(), timeout_fault_seconds=0.01, sleep=lambda s: None
        ),
    )
    session.execute(text("UPDATE eval.attempts SET usage = jsonb_set(usage, '{model_calls}', '6')"))

    assert model_calls_per_run(session.connection(), pilot_id) == {
        "correct": {"attempts": 5, "mean_calls": 6.0}
    }
    main = load_plan("main")
    estimate = estimate_quota(
        session.connection(), pilot_id, main, planned_runs(session.connection(), main)
    )

    # The pilot did not use Flash-Lite, so the pilot's overall mean (6) stands in: 150 runs.
    lite = estimate["gemini-3.5-flash-lite"]
    assert (lite["planned_runs"], lite["pilot_mean_calls_per_run"]) == (150, 6.0)
    assert (lite["requests"], lite["requests_with_headroom"]) == (900, 1125)
