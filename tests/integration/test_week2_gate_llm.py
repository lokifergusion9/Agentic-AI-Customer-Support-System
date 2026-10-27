"""Week 2 gate with the real model (opt-in: `uv run pytest -m llm`).

The headphones refund example must produce a schema-valid proposal with evidence refs and
policy citations that the business-rule validator accepts, and every model call must appear in
the audit trail with its token usage. Uses the free-tier quota (roughly 8-12 model calls) and
the real embedding model.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.agents.config import AgentConfig, build_provider
from app.config import get_settings
from app.policies.documents import load_policies
from app.policies.embeddings import get_embedder
from app.schemas.tickets import TicketCreate
from app.services.providers import ProviderClient
from app.services.tickets import create_ticket
from app.workflows.runner import WorkflowServices, start_resolution
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from scripts.ingest_policies import ingest
from scripts.seed_data import push_mock_payments, push_mock_shipping, seed
from tests.integration.test_proposals import ALICE, TODAY
from tests.integration.test_workflow import HEADPHONES

pytestmark = pytest.mark.llm


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    settings = get_settings()
    if not settings.llm_model or not settings.llm_api_key.get_secret_value():
        pytest.skip("set LLM_MODEL and LLM_API_KEY in .env to run the real-model gate")
    seed(conn, TODAY)
    ingest(conn, get_embedder(), load_policies())
    with Session(bind=conn, join_transaction_mode="create_savepoint") as session:
        yield session


@pytest.mark.parametrize("architecture", ["specialist", "single"])
def test_headphones_refund_with_the_real_model(db: Session, architecture: str) -> None:
    config = AgentConfig.from_settings()
    config = AgentConfig(**{**config.__dict__, "architecture": architecture})
    with (
        TestClient(
            payments_app, headers={"X-API-Key": payments_settings.mock_payments_api_key}
        ) as pay,
        TestClient(
            shipping_app, headers={"X-API-Key": shipping_settings.mock_shipping_api_key}
        ) as ship,
    ):
        push_mock_payments(pay, TODAY)
        push_mock_shipping(ship, TODAY)
        services = WorkflowServices(
            provider=build_provider(config),
            config=config,
            payments=ProviderClient("payment", pay),
            shipping=ProviderClient("shipping", ship),
            embedder=get_embedder(),
            checkpointer=InMemorySaver(),
            today=TODAY,
        )
        ticket = create_ticket(db, ALICE, TicketCreate(message=HEADPHONES, order_id="ord_1001"))
        run = start_resolution(db, ALICE, ticket.id, services)

    problems = db.execute(
        text(
            "SELECT event_type, redacted_payload FROM support.audit_events WHERE run_id = :r "
            "AND event_type IN ('run_halted', 'model_output_invalid', 'proposal_validated', "
            "'tool_rejected') ORDER BY id"
        ),
        {"r": run.id},
    ).all()
    assert run.status == "waiting_for_approval", (run.output, [tuple(p) for p in problems])
    proposal = db.execute(
        text("SELECT payload FROM support.action_proposals WHERE id = :p"),
        {"p": run.output["proposal_id"]},
    ).scalar_one()
    assert proposal["amount_minor"] == 249900
    assert proposal["evidence_refs"]
    assert any(ref.startswith("refund_policy_v1:") for ref in proposal["policy_refs"])

    calls = (
        db.execute(
            text(
                "SELECT redacted_payload FROM support.audit_events "
                "WHERE run_id = :r AND event_type = 'model_call'"
            ),
            {"r": run.id},
        )
        .scalars()
        .all()
    )
    assert calls and all(c["outcome"] == "ok" or c["retryable"] for c in calls)
    assert all(c["input_tokens"] is not None for c in calls if c["outcome"] == "ok")
