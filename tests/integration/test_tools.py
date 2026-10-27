"""Typed tools and the tool gateway: allowlists, validation, identity, audit (W2.4, W2.5)."""

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.policies.documents import load_policies
from app.schemas.tickets import TicketCreate
from app.services.providers import ProviderClient
from app.services.runs import start_run
from app.services.tickets import create_ticket
from app.tools.base import ToolContext
from app.tools.gateway import ToolGateway
from app.tools.registry import MODEL_TOOLS, READ_TOOLS, TOOLS
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from scripts.ingest_policies import ingest
from scripts.seed_data import push_mock_payments, push_mock_shipping, seed
from tests.fakes import FakeEmbedder
from tests.integration.test_proposals import ALICE, TODAY

AGENT = "agent:test"


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    seed(conn, TODAY)
    ingest(conn, FakeEmbedder(), load_policies())
    # A savepoint per session, so the gateway's rollback after a refusal keeps the seed data.
    with Session(bind=conn, join_transaction_mode="create_savepoint") as session:
        yield session


@pytest.fixture
def providers() -> Iterator[tuple[ProviderClient, ProviderClient]]:
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
        yield ProviderClient("payment", pay), ProviderClient("shipping", ship)


@pytest.fixture
def ctx(db: Session, providers: tuple[ProviderClient, ProviderClient]) -> ToolContext:
    ticket = create_ticket(
        db, ALICE, TicketCreate(message="Where is my refund?", order_id="ord_1001")
    )
    run = start_run(db, ALICE, ticket.id)
    payments, shipping = providers
    return ToolContext(db, ALICE, run.id, ticket.id, payments, shipping, FakeEmbedder(), TODAY)


def gateway(ctx: ToolContext, allowed: frozenset[str] = MODEL_TOOLS) -> ToolGateway:
    return ToolGateway(ctx, allowed, actor=AGENT)


def tool_events(ctx: ToolContext) -> list[tuple[str, dict[str, Any]]]:
    rows = ctx.db.execute(
        text(
            "SELECT event_type, redacted_payload FROM support.audit_events "
            "WHERE run_id = :run AND event_type LIKE 'tool_%' ORDER BY id"
        ),
        {"run": ctx.run_id},
    ).all()
    return [(row.event_type, row.redacted_payload) for row in rows]


def proposals(ctx: ToolContext) -> int:
    return ctx.db.execute(
        text("SELECT count(*) FROM support.action_proposals WHERE run_id = :run"),
        {"run": ctx.run_id},
    ).scalar_one()


def refund_operations(ctx: ToolContext, order_id: str) -> int:
    return ctx.db.execute(
        text("SELECT count(*) FROM support.refund_operations WHERE order_id = :order"),
        {"order": order_id},
    ).scalar_one()


# --- Registry --------------------------------------------------------------------------


def test_execute_tools_are_never_offered_to_the_model() -> None:
    assert {"execute_approved_refund", "update_ticket"}.isdisjoint(MODEL_TOOLS)
    assert "create_refund_proposal" in MODEL_TOOLS and "create_refund_proposal" not in READ_TOOLS


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_no_tool_accepts_identity_or_approval_arguments(name: str) -> None:
    properties = TOOLS[name].json_schema().get("properties", {})
    forbidden = {"customer_id", "reviewer_id", "approval_status", "approved", "api_key", "role"}
    assert forbidden.isdisjoint(properties)
    assert TOOLS[name].json_schema()["additionalProperties"] is False


# --- Read tools ------------------------------------------------------------------------


def test_read_tool_returns_data_and_is_audited(ctx: ToolContext) -> None:
    result = gateway(ctx).execute("get_order", {"order_id": "ord_1001"}, call_id="c1")

    assert result["success"] and result["data"]["id"] == "ord_1001"
    events = tool_events(ctx)
    assert [e for e, _ in events] == ["tool_requested", "tool_finished"]
    assert events[0][1] == {"tool": "get_order", "call_id": "c1", "args": {"order_id": "ord_1001"}}
    assert events[1][1]["status"] == "ok" and events[1][1]["duration_ms"] >= 0


def test_another_customers_order_is_refused_without_details(ctx: ToolContext) -> None:
    other = ctx.db.execute(
        text("SELECT id FROM support.orders WHERE customer_id <> 'cus_alice' LIMIT 1")
    ).scalar_one()

    result = gateway(ctx).execute("get_return_status", {"order_id": other}, call_id="c1")

    assert result == {
        "success": False,
        "error_code": "ORDER_NOT_FOUND",
        "retryable": False,
        "message": result["message"],
    }
    assert other not in result["message"]
    assert [e for e, _ in tool_events(ctx)] == ["tool_requested", "tool_rejected"]


def test_model_supplied_customer_id_is_rejected(ctx: ToolContext) -> None:
    result = gateway(ctx).execute(
        "get_order", {"order_id": "ord_1001", "customer_id": "cus_bob"}, call_id="c1"
    )
    assert result["error_code"] == "INVALID_ARGUMENTS"
    assert "customer_id" in result["message"]
    assert tool_events(ctx)[-1][1]["error_code"] == "INVALID_ARGUMENTS"


def test_arguments_may_arrive_as_a_json_string(ctx: ToolContext) -> None:
    result = gateway(ctx).execute("get_order", '{"order_id": "ord_1001"}', call_id="c1")
    assert result["success"]


def test_malformed_json_is_rejected(ctx: ToolContext) -> None:
    result = gateway(ctx).execute("get_order", "{order_id: ord_1001", call_id="c1")
    assert result["error_code"] == "INVALID_ARGUMENTS"


def test_tool_outside_the_allowlist_is_refused(ctx: ToolContext) -> None:
    investigation = gateway(ctx, READ_TOOLS)
    result = investigation.execute(
        "create_refund_proposal",
        {"order_id": "ord_1001", "amount_minor": 249900, "reason": "Return received"},
        call_id="c1",
    )
    assert result["error_code"] == "TOOL_NOT_ALLOWED"
    assert proposals(ctx) == 0


def test_model_cannot_call_refund_execution(ctx: ToolContext) -> None:
    result = gateway(ctx).execute("execute_approved_refund", {"proposal_id": "prp_x"}, call_id="c1")
    assert result["error_code"] == "TOOL_NOT_ALLOWED"


def test_unknown_tool_is_refused(ctx: ToolContext) -> None:
    result = gateway(ctx).execute("delete_all_orders", {}, call_id="c1")
    assert result["error_code"] == "TOOL_NOT_ALLOWED"


def test_unreachable_provider_is_a_retryable_error(ctx: ToolContext) -> None:
    ctx.payments = ProviderClient(
        "payment", httpx.Client(base_url="http://127.0.0.1:9", timeout=0.5)
    )
    result = gateway(ctx).execute("get_payment_status", {"order_id": "ord_1001"}, call_id="c1")
    # Refused connections are UNAVAILABLE; on Windows a closed port can time out instead.
    assert result["error_code"] in {"PROVIDER_UNAVAILABLE", "PROVIDER_TIMEOUT"}
    assert result["retryable"]
    finished = tool_events(ctx)[-1]
    assert finished[0] == "tool_finished" and finished[1]["status"] == "error"


def test_search_policy_query_text_is_not_audited(ctx: ToolContext) -> None:
    query = "refund after return received and inspection passed"
    result = gateway(ctx).execute(
        "search_policy",
        {"query": query, "policy_keys": ["refund_policy"], "effective_on": "2026-10-20"},
        call_id="c1",
    )
    assert result["success"] and result["data"]["status"] == "found"
    requested, finished = tool_events(ctx)
    assert requested[1]["args"]["query"] == {"chars": len(query)}
    assert finished[1]["citations"] == result["data"]["citations"]


# --- Proposal tool ---------------------------------------------------------------------


def test_eligible_refund_is_proposed_but_no_money_moves(ctx: ToolContext) -> None:
    result = gateway(ctx).execute(
        "create_refund_proposal",
        # Providers send JSON numbers as floats.
        {
            "order_id": "ord_1001",
            "amount_minor": 249900.0,
            "reason": "Return received, inspection passed",
            "evidence_refs": ["ret_1001"],
        },
        call_id="c1",
    )

    assert result["success"]
    assert result["data"]["payload"]["amount_minor"] == 249900
    assert result["data"]["status"] == "pending_approval"
    assert refund_operations(ctx, "ord_1001") == 0
    status = ctx.db.execute(
        text("SELECT status FROM support.workflow_runs WHERE id = :run"), {"run": ctx.run_id}
    ).scalar_one()
    assert status == "waiting_for_approval"
    assert ctx.payments.get("/payments/ord_1001")["refunds"] == []


def test_amount_other_than_the_allowed_one_is_refused(ctx: ToolContext) -> None:
    result = gateway(ctx).execute(
        "create_refund_proposal",
        {"order_id": "ord_1001", "amount_minor": 299900, "reason": "Customer asked for more"},
        call_id="c1",
    )
    assert result["error_code"] in {"AMOUNT_MISMATCH", "NOT_ELIGIBLE"}
    assert proposals(ctx) == 0


def test_lower_amount_within_the_balance_is_proposed_as_asked(ctx: ToolContext) -> None:
    # Eligibility (W1.14) honors a requested amount up to the maximum, so a lower amount is a
    # valid proposal; the reviewer sees it, and the W4 outcome evaluator catches a wrong one.
    result = gateway(ctx).execute(
        "create_refund_proposal",
        {"order_id": "ord_1001", "amount_minor": 24990, "reason": "Return received"},
        call_id="c1",
    )
    assert result["success"] and result["data"]["payload"]["amount_minor"] == 24990


def test_ineligible_order_is_refused(ctx: ToolContext) -> None:
    # ord_1002 is Alice's order that was already refunded in full.
    result = gateway(ctx).execute(
        "create_refund_proposal",
        {"order_id": "ord_1002", "amount_minor": 100, "reason": "Refund please"},
        call_id="c1",
    )
    assert result["error_code"] == "NOT_ELIGIBLE"
    assert proposals(ctx) == 0
    requested, rejected = tool_events(ctx)
    assert requested[1]["args"]["reason"] == {"chars": len("Refund please")}
    assert (
        rejected == ("tool_rejected", rejected[1]) and rejected[1]["error_code"] == "NOT_ELIGIBLE"
    )
