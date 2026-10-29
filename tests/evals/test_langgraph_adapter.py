"""The LangGraph adapter runs the real workflow on a task; configs are immutable rows (W4.7)."""

import pytest
from sqlalchemy import Connection, text

from app.agents.config import AgentConfig
from app.policies.documents import load_policies
from evals.adapters.langgraph import LangGraphAdapter
from evals.configs import agent_config, config_row, register_agent_config
from evals.db import EVAL_SCHEMA
from scripts.ingest_policies import ingest
from tests.fakes import FakeEmbedder, ScriptedModel
from tests.integration.test_workflow import headphones_scenario

FAKE = AgentConfig(
    provider="gemini", model_id="fake-model", prompt_version="v1", architecture="specialist"
)
ROW = config_row("fake_specialist_v1", FAKE)


@pytest.fixture
def policies(conn: Connection) -> None:
    # The policy agent cites search results; the eval environment never touches policy tables.
    ingest(conn, FakeEmbedder(), load_policies())


@pytest.fixture
def adapter() -> LangGraphAdapter:
    # The headphones script, with the fixture's return and payment IDs as evidence.
    return LangGraphAdapter(
        lambda config: ScriptedModel(headphones_scenario("ret_1001", "pay_1001"))
    )


def test_an_approved_refund_runs_end_to_end(policies, adapter, run_attempt) -> None:
    done = run_attempt("refund_eligible_001", adapter, agent_config=ROW)

    assert done.statuses() == ("completed", "resolved")
    assert done.result.stop_reason == "model_final"
    assert [r["amount_minor"] for r in done.refunds("ord_1001")] == [249900]
    assert "ord_1001" in done.result.final_response
    approvals = done.result.metadata["simulated_approvals"]
    assert [(a["outcome"], a["reviewer_id"]) for a in approvals] == [("approved", "sim|reviewer")]
    assert done.events("run_configured")[0]["config_id"] == FAKE.config_id


def test_a_rejected_refund_follows_the_rejection_path(policies, adapter, run_attempt) -> None:
    done = run_attempt("refund_eligible_002", adapter, agent_config=ROW)
    assert done.statuses() == ("completed", "resolved")
    assert done.result.metadata["route"] == "rejected"
    assert done.refunds("ord_1001") == []


def test_expiring_approvals_end_in_escalation(policies, adapter, run_attempt) -> None:
    done = run_attempt("refund_eligible_003", adapter, agent_config=ROW)
    assert done.statuses() == ("escalated", "escalated")
    assert [a["outcome"] for a in done.result.metadata["simulated_approvals"]] == ["expired"] * 3


def test_the_task_limits_cap_the_workflow_steps(policies, adapter, run_attempt) -> None:
    # Model calls and tool calls both take a step: 12 turns + 20 tool calls, capped at 25.
    done = run_attempt("refund_eligible_001", adapter, agent_config=ROW)
    assert done.result.metadata["max_steps"] == 25
    lookup = run_attempt("lookup_delivered_001", adapter, agent_config=ROW)
    assert lookup.result.metadata["max_steps"] == 20  # 8 turns + 12 tool calls


def test_a_config_row_round_trips_to_the_workflow_config() -> None:
    assert agent_config(ROW) == FAKE
    assert ROW["prompt_hashes"].keys() == {"classifier", "investigation", "policy", "resolution"}
    assert config_row("fake_specialist_v1", FAKE) == ROW  # deterministic


def test_registering_is_idempotent_and_configs_are_immutable(conn: Connection) -> None:
    register_agent_config(conn, ROW)
    register_agent_config(conn, ROW)
    count = conn.execute(text(f"SELECT count(*) FROM {EVAL_SCHEMA}.agent_configs")).scalar()
    prompts = conn.execute(text(f"SELECT id FROM {EVAL_SCHEMA}.prompt_versions")).scalars()
    assert count == 1
    assert set(prompts) == {"classifier_v1", "investigation_v1", "policy_v1", "resolution_v1"}

    changed = config_row("fake_specialist_v1", AgentConfig(**{**FAKE.__dict__, "max_steps": 10}))
    with pytest.raises(ValueError, match="immutable"):
        register_agent_config(conn, changed)


@pytest.mark.llm
def test_the_real_agent_resolves_the_reference_refund(policies, run_attempt) -> None:
    """Calls Gemini: `uv run pytest -m llm tests/evals/test_langgraph_adapter.py`."""
    config = AgentConfig.from_settings()
    done = run_attempt(
        "refund_eligible_001",
        LangGraphAdapter(),
        agent_config=config_row(config.config_id, config),
    )
    assert done.statuses()[0] in ("completed", "escalated")
    assert done.events("model_call")
