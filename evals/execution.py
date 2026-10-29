"""The `eval_attempt` job: one attempt of one run (W4.8, E§17).

reset environment -> run adapter -> snapshot -> evaluate -> store scores.

- Execution status (how the attempt ran) and evaluation status (how it scored) are separate.
  A confident wrong answer is `completed` + `fail`; a turn-limit stop is `completed` with its
  stop reason kept.
- An infrastructure failure (an exception out of the environment or the adapter) is retried
  by the queue as a new attempt in a fresh environment, a bounded number of times. Agent
  behaviour is never retried: a finished attempt is final, whatever its score.
- The model provider failing is not the agent's behaviour either. A quota error (HTTP 429)
  defers the job to when the quota is back, without using a retry, and holds back the other
  queued runs of that model; other provider errors follow the infrastructure retry policy.
  A long experiment therefore waits out a daily quota and resumes, losing no run (W5.2).
- A worker that dies leaves its attempt `running`. Whoever holds the job next marks it
  `interrupted` and starts a replacement; only the final attempt is counted. A stale worker
  cannot finalize: every final write checks the job's lease token and the attempt's status.
"""

import asyncio
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SCHEMA
from app.jobs.handlers import HandlerContext, register
from app.jobs.queue import Deferred, Job
from app.policies.embeddings import Embedder
from app.services.providers import ProviderClient
from evals.adapters.base import AgentAdapter, AgentResult, AttemptContext
from evals.adapters.langgraph import LangGraphAdapter
from evals.adapters.scripted import ScriptedAdapter
from evals.db import EVAL_SCHEMA
from evals.environment import MockClients, capture, environment_lock, prepare
from evals.evaluators.base import AttemptRecord
from evals.evaluators.suite import Verdict, evaluate, store_scores
from evals.experiments import JOB_TYPE
from evals.reviewer_simulator import ReviewerSimulator
from evals.task_format import AgentView, EvaluatorView

FINAL_EXECUTION = ("completed", "timed_out", "budget_exceeded", "infrastructure_error", "cancelled")
# Stop reasons that mean the run ran out of budget rather than finishing (Q4).
BUDGET_STOPS = {"token_limit"}


class StaleAttempt(Exception):
    """This worker lost the job, or the attempt was replaced; it must not finalize."""


def default_adapter(config: dict[str, Any]) -> AgentAdapter:
    if config["adapter"] == "scripted":
        return ScriptedAdapter(config["model_id"])
    return LangGraphAdapter()


@dataclass
class EvalContext:
    mocks: MockClients
    embedder: Embedder
    # Builds the adapter for an agent_configs row; tests pass scripted models.
    adapter_for: Callable[[dict[str, Any]], AgentAdapter] = default_adapter
    timeout_fault_seconds: float | None = None
    sleep: Callable[[float], None] = time.sleep


def install(ctx: EvalContext) -> None:
    """Register the eval_attempt handler with the job worker."""

    @register(JOB_TYPE)
    def handle(db: Session, job: Job, _: HandlerContext | None) -> None:
        run_attempt_job(db, job, ctx)


def run_attempt_job(db: Session, job: Job, ctx: EvalContext) -> None:
    run = load_run(db, job.payload["run_id"])
    if run["execution_status"] in FINAL_EXECUTION:
        return  # already final: a repeated job changes nothing
    if run["experiment_status"] == "cancelled":
        finish_run(db, job, run["id"], "cancelled", "not_applicable")
        return
    interrupt_stale_attempts(db, run["id"])

    task = EvaluatorView.model_validate(run["evaluator_view"])
    agent = AgentView.model_validate(run["agent_view"])
    connection = db.connection()
    attempt_id: str | None = None
    try:
        with environment_lock(connection):
            env = prepare(
                connection, ctx.mocks, task, timeout_fault_seconds=ctx.timeout_fault_seconds
            )
            attempt_id = start_attempt(db, run, env.environment_id)

            reviewer = ReviewerSimulator(task.reviewer_behavior)
            context = AttemptContext(
                db=db,
                environment=env,
                payments=ProviderClient("payment", ctx.mocks.payments),
                shipping=ProviderClient("shipping", ctx.mocks.shipping),
                ticketing=ProviderClient("ticketing", ctx.mocks.ticketing),
                embedder=ctx.embedder,
                reviewer=reviewer,
                limits=task.evaluation.limits,
                agent_config=run["agent_config"],
                sleep=ctx.sleep,
            )
            result, active_ms, timed_out = run_with_timeout(
                ctx.adapter_for(run["agent_config"]), agent, context
            )
            workflow_run_id = result.metadata.get("workflow_run_id")
            state = capture(connection, ctx.mocks, env, attempt_id, workflow_run_id)
            db.commit()
    except StaleAttempt:
        raise
    except Exception as error:  # noqa: BLE001 - anything out of the environment or adapter
        db.rollback()
        record_infrastructure_error(db, job, run["id"], attempt_id, error)
        return

    trace = read_trace(db, attempt_id)
    if (failure := provider_failure(trace)) is not None:
        end_attempt_on_provider_failure(db, job, run, attempt_id, failure)
        return
    record = AttemptRecord(
        task=task,
        agent_input=agent.agent_input,
        snapshot=state,
        trace=trace,
        final_response=result.final_response,
        stop_reason=result.stop_reason,
        workflow_run_id=workflow_run_id,
    )
    verdict = evaluate(record)
    execution = execution_status(result, timed_out)
    finalize(
        db,
        job,
        run,
        attempt_id,
        result=result,
        execution=execution,
        verdict=verdict,
        active_ms=active_ms - reviewer.total_wait_ms,
        approval_wait_ms=reviewer.total_wait_ms,
        usage=usage(trace),
    )


# --- Steps --------------------------------------------------------------------------------------


def load_run(db: Session, run_id: str) -> dict[str, Any]:
    row = db.execute(
        text(
            "SELECT r.id, r.experiment_id, r.execution_status, e.status AS experiment_status, "
            "t.category, t.agent_view, t.evaluator_view, to_jsonb(c) AS agent_config "
            f"FROM {EVAL_SCHEMA}.runs r "
            f"JOIN {EVAL_SCHEMA}.experiments e ON e.id = r.experiment_id "
            f"JOIN {EVAL_SCHEMA}.task_versions t ON t.id = r.task_version_id "
            f"JOIN {EVAL_SCHEMA}.agent_configs c ON c.id = r.agent_config_id "
            "WHERE r.id = :id"
        ),
        {"id": run_id},
    ).one()
    return dict(row._mapping)


def interrupt_stale_attempts(db: Session, run_id: str) -> None:
    """An attempt still `running` while this worker holds the job belongs to a dead worker."""
    db.execute(
        text(
            f"UPDATE {EVAL_SCHEMA}.attempts SET status = 'interrupted', finished_at = now(), "
            "stop_reason = 'worker_lost' WHERE run_id = :run AND status = 'running'"
        ),
        {"run": run_id},
    )
    db.commit()


def start_attempt(db: Session, run: dict[str, Any], environment_id: str) -> str:
    number = db.execute(
        text(
            f"SELECT COALESCE(max(attempt_number), 0) + 1 FROM {EVAL_SCHEMA}.attempts WHERE run_id = :run"
        ),
        {"run": run["id"]},
    ).scalar_one()
    attempt_id = f"att_{run['id'].removeprefix('erun_')}_{number}"
    db.execute(
        text(
            f"INSERT INTO {EVAL_SCHEMA}.attempts (id, run_id, attempt_number, environment_id) "
            "VALUES (:id, :run, :number, :env)"
        ),
        {"id": attempt_id, "run": run["id"], "number": number, "env": environment_id},
    )
    db.execute(
        text(
            f"UPDATE {EVAL_SCHEMA}.runs SET execution_status = 'running', updated_at = now() "
            "WHERE id = :run"
        ),
        {"run": run["id"]},
    )
    db.execute(
        text(
            f"UPDATE {EVAL_SCHEMA}.experiments SET status = 'running', started_at = now() "
            "WHERE id = :id AND status = 'queued'"
        ),
        {"id": run["experiment_id"]},
    )
    db.commit()
    return attempt_id


def run_with_timeout(
    adapter: AgentAdapter, agent: AgentView, context: AttemptContext
) -> tuple[AgentResult, float, bool]:
    """Run the adapter; at the task's timeout no new tool call may start (cancel is set)."""
    timer = threading.Timer(context.limits.timeout_seconds, context.cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        result = asyncio.run(adapter.run(agent, context))
    finally:
        timer.cancel()
    elapsed_ms = (time.monotonic() - started) * 1000
    return result, elapsed_ms, context.cancel.is_set()


PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini's daily quotas reset at midnight Pacific
MIN_QUOTA_WAIT_SECONDS = 60


def provider_failure(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The model provider's error that ended the run, if one did: not agent behaviour."""
    halts = [
        e["payload"]
        for e in trace
        if e["event_type"] == "run_halted" and e["payload"].get("stop_reason") == "model_error"
    ]
    if not halts:
        return None
    last_error = next(
        (
            e["payload"]
            for e in reversed(trace)
            if e["event_type"] == "model_call" and e["payload"].get("outcome") == "error"
        ),
        {},
    )
    return {
        "code": halts[-1].get("code"),
        "quota": last_error.get("error_detail"),
        "retry_after_s": last_error.get("retry_after_s"),
    }


def quota_resume_at(failure: dict[str, Any], now: datetime | None = None) -> datetime:
    """When to try again: past the next Pacific midnight for a daily quota, else soon."""
    now = now or datetime.now(UTC)
    if "PerDay" in (failure.get("quota") or ""):
        tomorrow = now.astimezone(PACIFIC) + timedelta(days=1)
        return tomorrow.replace(hour=0, minute=5, second=0, microsecond=0).astimezone(UTC)
    wait = max(failure.get("retry_after_s") or 0, MIN_QUOTA_WAIT_SECONDS)
    return now + timedelta(seconds=wait)


def end_attempt_on_provider_failure(
    db: Session, job: Job, run: dict[str, Any], attempt_id: str, failure: dict[str, Any]
) -> None:
    detail = {"provider_error": failure["code"], "quota": failure["quota"]}
    db.execute(
        text(
            f"UPDATE {EVAL_SCHEMA}.attempts SET status = 'error', finished_at = now(), "
            "execution_status = 'infrastructure_error', error = CAST(:error AS jsonb) "
            "WHERE id = :id AND status = 'running'"
        ),
        {"id": attempt_id, "error": json.dumps(detail)},
    )
    db.commit()
    if failure["code"] != "HTTP_429":
        record_infrastructure_error(
            db, job, run["id"], None, RuntimeError(f"model provider error {failure['code']}")
        )
        return
    resume = quota_resume_at(failure)
    # Other queued runs of this model would only meet the same quota: hold them back too.
    db.execute(
        text(
            f"UPDATE {SCHEMA}.jobs j SET available_at = :at "
            f"FROM {EVAL_SCHEMA}.runs r JOIN {EVAL_SCHEMA}.agent_configs c "
            "ON c.id = r.agent_config_id "
            "WHERE j.job_type = 'eval_attempt' AND j.status = 'queued' "
            "AND j.payload->>'run_id' = r.id AND c.model_id = :model AND j.available_at < :at"
        ),
        {"at": resume, "model": run["agent_config"]["model_id"]},
    )
    db.execute(
        text(f"UPDATE {EVAL_SCHEMA}.runs SET execution_status = 'queued' WHERE id = :id"),
        {"id": run["id"]},
    )
    db.commit()
    raise Deferred(resume, f"model quota exhausted ({failure['quota'] or failure['code']})")


def execution_status(result: AgentResult, timed_out: bool) -> str:
    if timed_out:
        return "timed_out"
    if result.stop_reason == "cancelled":
        return "cancelled"
    if result.stop_reason in BUDGET_STOPS:
        return "budget_exceeded"
    return "completed"  # including turn-limit stops, whose stop reason is kept


def read_trace(db: Session, attempt_id: str) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            f"SELECT sequence_number, actor, event_type, payload FROM {EVAL_SCHEMA}.trace_events "
            "WHERE attempt_id = :a ORDER BY sequence_number"
        ),
        {"a": attempt_id},
    )
    return [dict(row._mapping) for row in rows]


def usage(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokens over every model call, retries included. A field any call did not report is
    unknown (null), never 0 (W2.16)."""
    calls = [e["payload"] for e in trace if e["event_type"] == "model_call"]
    totals: dict[str, Any] = {"model_calls": len(calls)}
    for name in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
        values = [c.get(name) for c in calls if c.get("outcome") == "ok"]
        totals[name] = None if any(v is None for v in values) else sum(values)
    totals["retried_calls"] = sum(1 for c in calls if c.get("outcome") != "ok")
    return totals


def finalize(
    db: Session,
    job: Job,
    run: dict[str, Any],
    attempt_id: str,
    *,
    result: AgentResult,
    execution: str,
    verdict: Verdict,
    active_ms: float,
    approval_wait_ms: float,
    usage: dict[str, Any],
) -> None:
    """Write the attempt's outcome and scores in one transaction, only while still ours."""
    hold_lease(db, job)
    updated = db.execute(
        text(
            f"UPDATE {EVAL_SCHEMA}.attempts SET status = 'finished', finished_at = now(), "
            "execution_status = :execution, workflow_run_id = :wr, stop_reason = :stop, "
            "final_response = :response, active_ms = :active, approval_wait_ms = :wait, "
            "usage = CAST(:usage AS jsonb), verified_success = :verified, "
            "error = CAST(:error AS jsonb) "
            "WHERE id = :id AND status = 'running' RETURNING id"
        ),
        {
            "id": attempt_id,
            "execution": execution,
            "wr": result.metadata.get("workflow_run_id"),
            "stop": result.stop_reason,
            "response": result.final_response,
            "active": round(active_ms),
            "wait": round(approval_wait_ms),
            "usage": json.dumps(usage),
            "verified": verdict.verified_success,
            # An agent run that ended in an unexpected error keeps the error class (not retried).
            "error": json.dumps({"error": result.metadata["error"]})
            if result.metadata.get("error")
            else None,
        },
    ).scalar_one_or_none()
    if updated is None:
        db.rollback()
        raise StaleAttempt(attempt_id)
    store_scores(db.connection(), attempt_id, run["experiment_id"], run["category"], verdict)
    set_run(db, run, execution, verdict.status)
    db.commit()


def record_infrastructure_error(
    db: Session, job: Job, run_id: str, attempt_id: str | None, error: Exception
) -> None:
    """Mark the attempt; retry in a fresh environment if the job has attempts left."""
    detail = {"error": f"{type(error).__name__}: {error}"}
    if attempt_id is not None:
        db.execute(
            text(
                f"UPDATE {EVAL_SCHEMA}.attempts SET status = 'error', finished_at = now(), "
                "execution_status = 'infrastructure_error', error = CAST(:error AS jsonb) "
                "WHERE id = :id AND status = 'running'"
            ),
            {"id": attempt_id, "error": json.dumps(detail)},
        )
        db.commit()
    if job.attempts < job.max_attempts:
        raise RuntimeError(f"infrastructure error, retrying: {detail['error']}") from error
    finish_run(db, job, run_id, "infrastructure_error", "not_applicable")


def finish_run(db: Session, job: Job, run_id: str, execution: str, evaluation: str) -> None:
    hold_lease(db, job)
    run = load_run(db, run_id)
    set_run(db, run, execution, evaluation)
    db.commit()


def set_run(db: Session, run: dict[str, Any], execution: str, evaluation: str) -> None:
    db.execute(
        text(
            f"UPDATE {EVAL_SCHEMA}.runs SET execution_status = :execution, "
            "evaluation_status = :evaluation, updated_at = now() WHERE id = :id"
        ),
        {"id": run["id"], "execution": execution, "evaluation": evaluation},
    )
    db.execute(
        text(
            f"UPDATE {EVAL_SCHEMA}.experiments SET status = 'completed', completed_at = now() "
            "WHERE id = :id AND status IN ('queued', 'running') AND NOT EXISTS ("
            f"SELECT 1 FROM {EVAL_SCHEMA}.runs WHERE experiment_id = :id "
            "AND execution_status IN ('queued', 'running'))"
        ),
        {"id": run["experiment_id"]},
    )


def hold_lease(db: Session, job: Job) -> None:
    """Lock the job row and check the lease is still ours, for the rest of the transaction."""
    held = db.execute(
        text(
            f"SELECT id FROM {SCHEMA}.jobs WHERE id = :id AND status = 'running' "
            "AND lease_token = :token FOR UPDATE"
        ),
        {"id": job.id, "token": job.lease_token},
    ).scalar_one_or_none()
    if held is None:
        db.rollback()
        raise StaleAttempt(job.id)
