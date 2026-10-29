"""Week 4 gate, part 2, with the real model (opt-in: `uv run pytest -m llm -s tests/evals/test_week4_gate_llm.py`).

The real agent (the model and architecture in .env) runs the 5 pilot tasks end to end through
the evaluation platform. Every attempt must finish, be scored, and leave a trace with its model
calls (with token usage) and tool calls. It does not assert that the agent is right: how
reliable it is, is what the platform measures. Uses free-tier quota (roughly 50 model calls)
and the real embedding model.
"""

import pytest
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.jobs import queue
from app.policies.documents import load_policies
from app.policies.embeddings import get_embedder
from evals.configs import named_config
from evals.execution import EvalContext, run_attempt_job
from evals.experiments import ExperimentPlan, create_experiment
from scripts.ingest_policies import ingest

pytestmark = pytest.mark.llm


@pytest.fixture
def session(conn: Connection) -> Session:
    settings = get_settings()
    if not settings.llm_model or not settings.llm_api_key.get_secret_value():
        pytest.skip("set LLM_MODEL and LLM_API_KEY in .env to run the real-model gate")
    ingest(conn, get_embedder(), load_policies())
    return Session(bind=conn, join_transaction_mode="create_savepoint")


def test_the_real_agent_runs_the_pilot_end_to_end_and_every_attempt_is_scored(session, mocks):
    ctx = EvalContext(mocks=mocks, embedder=get_embedder(), timeout_fault_seconds=0.01)
    plan = ExperimentPlan(
        "week4-gate", "support-v1", "pilot", [named_config("settings")], 1, "gate"
    )
    create_experiment(session, plan)

    while job := queue.claim(session, "gate", lease_seconds=900, job_types=["eval_attempt"]):
        run_attempt_job(session, job, ctx)
        queue.complete(session, job)

    rows = session.execute(
        text(
            "SELECT t.task_id, r.execution_status, r.evaluation_status, a.verified_success, "
            "a.stop_reason, a.usage, a.error, a.id AS attempt_id, "
            "(SELECT count(*) FROM eval.trace_events e WHERE e.attempt_id = a.id "
            "AND e.event_type = 'model_call') AS model_calls, "
            "(SELECT count(*) FROM eval.trace_events e WHERE e.attempt_id = a.id "
            "AND e.event_type = 'tool_requested') AS tool_calls, "
            "(SELECT count(*) FROM eval.scores s WHERE s.attempt_id = a.id) AS scores, "
            "(SELECT string_agg(s.evaluator_name || '.' || s.dimension || '=' || s.reason_code, ', ') "
            " FROM eval.scores s WHERE s.attempt_id = a.id AND s.status = 'fail') AS failures "
            "FROM eval.runs r JOIN eval.task_versions t ON t.id = r.task_version_id "
            "JOIN eval.attempts a ON a.run_id = r.id ORDER BY t.task_id"
        )
    ).all()

    print()
    for row in rows:
        print(
            f"{row.task_id:36} {row.execution_status}/{row.evaluation_status:7} "
            f"verified={row.verified_success} stop={row.stop_reason} "
            f"model_calls={row.model_calls} tools={row.tool_calls} tokens_in={row.usage['input_tokens']}"
        )
        if row.error:
            print(f"{'':36} error: {row.error}")
        if row.failures:
            print(f"{'':36} failed: {row.failures}")

    assert len(rows) == 5
    for row in rows:
        assert row.execution_status in ("completed", "budget_exceeded", "timed_out"), row.task_id
        assert row.evaluation_status in ("pass", "partial", "fail"), row.task_id
        assert row.scores > 20, row.task_id
        assert row.model_calls >= 1, row.task_id  # every model operation is in the trace
        assert row.tool_calls >= 1, row.task_id
    assert (
        session.execute(
            text("SELECT count(*) FROM eval.scores WHERE status = 'evaluation_error'")
        ).scalar()
        == 0
    )
