"""Experiments, the eval_attempt job and the worker rules around it (W4.8, W4.10)."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.jobs import queue
from app.jobs.queue import LeaseLost
from evals.adapters.scripted import ScriptedAdapter
from evals.configs import scripted_config_row
from evals.evaluators import suite
from evals.execution import EvalContext, StaleAttempt, lock_connection, run_attempt_job
from evals.experiments import ExperimentPlan, create_experiment, run_order
from tests.fakes import FakeEmbedder


@pytest.fixture
def session(conn: Connection) -> Session:
    return Session(bind=conn, join_transaction_mode="create_savepoint")


@pytest.fixture
def eval_ctx(mocks) -> EvalContext:
    return EvalContext(
        mocks=mocks,
        embedder=FakeEmbedder(),
        timeout_fault_seconds=0.01,
        sleep=lambda seconds: None,
    )


def plan(*scripts: str, key: str | None = "k1", split: str = "pilot") -> ExperimentPlan:
    return ExperimentPlan(
        name="t",
        dataset="support-v1",
        split=split,
        configs=[scripted_config_row(s) for s in scripts],
        repetitions=1,
        idempotency_key=key,
    )


def claim(session: Session, worker: str = "w1"):
    return queue.claim(session, worker, lease_seconds=600, job_types=["eval_attempt"])


def first_in_line(session: Session, job_id: str) -> None:
    """A retried or requeued job waits behind the others; the test wants this one next."""
    session.execute(
        text("UPDATE support.jobs SET available_at = now() - interval '1 day' WHERE id = :id"),
        {"id": job_id},
    )


def run_all(session: Session, ctx: EvalContext) -> int:
    done = 0
    while (job := claim(session)) is not None:
        run_attempt_job(session, job, ctx)
        queue.complete(session, job)
        done += 1
    return done


def runs(session: Session) -> list[tuple]:
    rows = session.execute(
        text(
            "SELECT t.task_id, c.id, r.execution_status, r.evaluation_status "
            "FROM eval.runs r JOIN eval.task_versions t ON t.id = r.task_version_id "
            "JOIN eval.agent_configs c ON c.id = r.agent_config_id ORDER BY t.task_id, c.id"
        )
    )
    return [tuple(r) for r in rows]


def count(session: Session, sql: str) -> int:
    return session.execute(text(sql)).scalar_one()


# --- Creating an experiment ---------------------------------------------------------------------


def test_experiment_runs_and_jobs_are_created_together(session) -> None:
    experiment_id, created = create_experiment(session, plan("correct", "false_success_claim"))

    assert created
    assert count(session, "SELECT count(*) FROM eval.runs") == 10
    jobs = session.execute(
        text(
            "SELECT count(*), count(DISTINCT dedupe_key) FROM support.jobs "
            "WHERE job_type = 'eval_attempt'"
        )
    ).one()
    assert tuple(jobs) == (10, 10)
    manifest = session.execute(
        text("SELECT manifest FROM eval.experiments WHERE id = :id"), {"id": experiment_id}
    ).scalar_one()
    assert set(manifest) >= {
        "dataset_version",
        "tasks",
        "configs",
        "limits",
        "evaluator_versions",
        "environment_version",
        "code_commit",
        "lockfile_hash",
        "repetitions",
    }
    assert len(manifest["tasks"]) == 5


def test_a_repeated_create_returns_the_same_experiment(session) -> None:
    first, created = create_experiment(session, plan("correct"))
    second, again = create_experiment(session, plan("correct"))
    assert (first, created, again) == (second, True, False)
    assert count(session, "SELECT count(*) FROM eval.runs") == 5


def test_configurations_are_interleaved_in_run_order() -> None:
    order = list(run_order(["t1", "t2", "t3"], ["a", "b"], 1))
    firsts = [config for i, (_, _, config) in enumerate(order) if i % 2 == 0]
    assert firsts == ["a", "b", "a"]  # the starting configuration rotates
    assert {(t, c) for t, _, c in order} == {(t, c) for t in ("t1", "t2", "t3") for c in "ab"}


def test_a_changed_agent_config_under_the_same_name_is_refused(session) -> None:
    create_experiment(session, plan("correct", key="a"))
    changed = {**scripted_config_row("correct"), "config_hash": "sha256:other"}
    with pytest.raises(ValueError, match="immutable"):
        create_experiment(session, ExperimentPlan("t", "support-v1", "pilot", [changed], 1, "b"))


# --- Running attempts ---------------------------------------------------------------------------


def test_the_correct_agent_passes_and_a_broken_one_completes_with_fail(session, eval_ctx) -> None:
    create_experiment(session, plan("correct", "false_success_claim"))
    assert run_all(session, eval_ctx) == 10

    table = {(t, c): (e, v) for t, c, e, v in runs(session)}
    assert table[("refund_eligible_001", "scripted_correct")] == ("completed", "pass")
    # A confident wrong answer: execution completed, evaluation fail (W4.8).
    assert table[("refund_eligible_001", "scripted_false_success_claim")] == ("completed", "fail")
    assert session.execute(text("SELECT status FROM eval.experiments")).scalar_one() == "completed"
    attempt = session.execute(
        text(
            "SELECT a.status, a.verified_success, a.approval_wait_ms, a.usage "
            "FROM eval.attempts a JOIN eval.runs r ON r.id = a.run_id "
            "WHERE r.agent_config_id = 'scripted_correct' "
            "AND r.task_version_id = 'refund_eligible_001@v1'"
        )
    ).one()
    assert (attempt.status, attempt.verified_success) == ("finished", True)
    assert attempt.approval_wait_ms is not None and attempt.usage["model_calls"] == 0
    assert count(session, "SELECT count(*) FROM eval.scores") > 50
    assert count(session, "SELECT count(*) FROM eval.trace_events") > 20


def test_a_finished_run_is_never_run_again(session, eval_ctx) -> None:
    create_experiment(session, plan("correct"))
    run_all(session, eval_ctx)
    session.execute(
        text(
            "UPDATE support.jobs SET status = 'queued', finished_at = NULL WHERE job_type = 'eval_attempt'"
        )
    )
    assert run_all(session, eval_ctx) == 5
    assert count(session, "SELECT max(attempt_number) FROM eval.attempts") == 1


def test_evaluator_failures_are_recorded_apart_from_the_agent(
    session, eval_ctx, monkeypatch
) -> None:
    def broken(record):
        raise KeyError("x")

    monkeypatch.setattr(suite, "EVALUATORS", [(n, v, broken) for n, v, _ in suite.EVALUATORS])
    create_experiment(session, plan("correct"))
    run_all(session, eval_ctx)
    assert {(e, v) for _, _, e, v in runs(session)} == {("completed", "evaluation_error")}


def test_a_cancelled_experiment_starts_nothing(session, eval_ctx) -> None:
    experiment_id, _ = create_experiment(session, plan("correct"))
    session.execute(
        text("UPDATE eval.experiments SET status = 'cancelled' WHERE id = :id"),
        {"id": experiment_id},
    )
    run_all(session, eval_ctx)
    assert {(e, v) for _, _, e, v in runs(session)} == {("cancelled", "not_applicable")}
    assert count(session, "SELECT count(*) FROM eval.attempts") == 0


# --- Retries and lost workers (W4.8, W4.10) -----------------------------------------------------


class Flaky:
    """Fails with an infrastructure error `times` times, then runs the correct script."""

    def __init__(self, times: int) -> None:
        self.times = times
        self.runs = 0

    def __call__(self, config):
        outer = self

        class Adapter:
            async def run(self, task_input, context):
                outer.runs += 1
                if outer.runs <= outer.times:
                    raise ConnectionError("mock payments unreachable")
                return await ScriptedAdapter("correct").run(task_input, context)

        return Adapter()


def flaky_ctx(mocks, times: int) -> EvalContext:
    return EvalContext(
        mocks=mocks,
        embedder=FakeEmbedder(),
        adapter_for=Flaky(times),
        timeout_fault_seconds=0.01,
        sleep=lambda seconds: None,
    )


def test_infrastructure_failure_is_retried_in_a_new_attempt_and_fresh_environment(
    session, mocks
) -> None:
    ctx = flaky_ctx(mocks, 1)
    create_experiment(session, plan("correct"))
    job = claim(session)
    with pytest.raises(RuntimeError, match="retrying"):
        run_attempt_job(session, job, ctx)
    queue.fail(session, job, "infrastructure error")
    first_in_line(session, job.id)
    retry = claim(session)
    assert retry.id == job.id and retry.attempts == 2
    run_attempt_job(session, retry, ctx)
    queue.complete(session, retry)

    attempts = session.execute(
        text(
            "SELECT attempt_number, status, execution_status, environment_id FROM eval.attempts "
            "WHERE run_id = :run ORDER BY attempt_number"
        ),
        {"run": job.payload["run_id"]},
    ).all()
    assert [(a.attempt_number, a.status, a.execution_status) for a in attempts] == [
        (1, "error", "infrastructure_error"),
        (2, "finished", "completed"),
    ]
    assert attempts[0].environment_id != attempts[1].environment_id


def test_agent_failures_are_never_retried(session, eval_ctx) -> None:
    create_experiment(session, plan("false_success_claim"))
    run_all(session, eval_ctx)
    assert count(session, "SELECT max(attempt_number) FROM eval.attempts") == 1
    assert count(session, "SELECT count(*) FROM support.jobs WHERE attempts > 1") == 0


def test_the_last_infrastructure_failure_ends_the_run(session, mocks) -> None:
    create_experiment(session, plan("correct"))
    session.execute(text("UPDATE support.jobs SET max_attempts = 1"))
    run_attempt_job(session, claim(session), flaky_ctx(mocks, 99))
    assert {(e, v) for _, _, e, v in runs(session)} >= {("infrastructure_error", "not_applicable")}


def test_a_killed_worker_leaves_an_interrupted_attempt_and_one_replacement(
    session, eval_ctx
) -> None:
    create_experiment(session, plan("correct"))
    dead = claim(session, "worker-dead")
    run_id = dead.payload["run_id"]
    # What the dead worker had written before it died.
    session.execute(
        text(
            "INSERT INTO eval.attempts (id, run_id, attempt_number, environment_id) "
            "VALUES ('att_dead', :run, 1, 'env_dead')"
        ),
        {"run": run_id},
    )
    session.execute(
        text("UPDATE eval.runs SET execution_status = 'running' WHERE id = :run"), {"run": run_id}
    )
    assert queue.requeue_expired(session, now=datetime.now(UTC) + timedelta(minutes=10))

    first_in_line(session, dead.id)
    replacement = claim(session, "worker-new")
    run_attempt_job(session, replacement, eval_ctx)
    queue.complete(session, replacement)

    attempts = session.execute(
        text(
            "SELECT attempt_number, status, stop_reason FROM eval.attempts "
            "WHERE run_id = :run ORDER BY attempt_number"
        ),
        {"run": run_id},
    ).all()
    assert [tuple(a) for a in attempts] == [
        (1, "interrupted", "worker_lost"),
        (2, "finished", "model_final"),
    ]
    with pytest.raises(LeaseLost):  # the dead worker cannot finish the job
        queue.complete(session, dead)
    done = "SELECT count(*) FROM eval.runs WHERE id = '{}' AND execution_status = 'completed'"
    assert count(session, done.format(run_id)) == 1


def test_a_stale_worker_cannot_finalize_an_attempt(session, eval_ctx) -> None:
    create_experiment(session, plan("correct"))
    stale = claim(session, "worker-stale")
    session.execute(
        text("UPDATE support.jobs SET lease_token = 'someone-else' WHERE id = :id"),
        {"id": stale.id},
    )
    with pytest.raises(StaleAttempt):
        run_attempt_job(session, stale, eval_ctx)
    assert count(session, "SELECT count(*) FROM eval.scores") == 0
    assert count(session, "SELECT count(*) FROM eval.attempts WHERE status = 'finished'") == 0


def test_two_workers_never_claim_the_same_eval_job(session) -> None:
    create_experiment(session, plan("correct"))
    claimed = []
    while (job := claim(session, f"w{len(claimed) % 2}")) is not None:
        claimed.append(job.id)
    assert len(claimed) == len(set(claimed)) == 5


def test_the_eval_worker_ignores_other_job_types(session) -> None:
    session.execute(
        text(
            "INSERT INTO support.jobs (id, job_type, payload) "
            "VALUES ('job_wf', 'workflow_run', '{}'::jsonb)"
        )
    )
    assert claim(session) is None


def test_the_environment_lock_survives_the_commits_of_a_real_engine_session(engine) -> None:
    """The worker's session owns its connection: a commit hands it back to the pool.

    The other tests bind the session to one connection, where a commit leaves it open, so they
    cannot see this. A lock held on the session's connection made every real attempt fail with
    "This Connection is closed" after the agent had finished.
    """
    with Session(engine) as db, lock_connection(db) as lock:
        assert lock is not db.connection()
        db.commit()
        assert db.connection() is not lock
        assert lock.execute(text("SELECT 1")).scalar_one() == 1  # still open after the commit


def test_a_session_bound_to_one_connection_shares_it_for_the_lock(session: Session, conn) -> None:
    with lock_connection(session) as lock:
        assert lock is conn
