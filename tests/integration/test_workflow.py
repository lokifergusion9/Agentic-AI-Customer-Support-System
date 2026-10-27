"""The resolution workflow end to end with a scripted model (W2.6–W2.15, Week 2 gate).

The scripted model answers per agent (recognized by its system prompt): first the tool calls
it plans, then a final turn, then its JSON. Everything else is real: tools, services, the mock
providers, the business-rule validator, the graph and its checkpoints.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.agents.config import AgentConfig
from app.policies.documents import load_policies
from app.schemas.tickets import TicketCreate
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient
from app.services.tickets import create_ticket, get_ticket
from app.workflows.checkpoints import postgres_checkpointer
from app.workflows.runner import WorkflowServices, resume_with_message, start_resolution
from mock_services.payments.main import app as payments_app
from mock_services.payments.main import settings as payments_settings
from mock_services.shipping.main import app as shipping_app
from mock_services.shipping.main import settings as shipping_settings
from scripts.ingest_policies import ingest
from scripts.seed_data import push_mock_payments, push_mock_shipping, seed
from tests.fakes import FakeCall, FakeEmbedder, ScriptedModel, json_reply, reply
from tests.integration.test_policy_search import REFUND_QUESTION
from tests.integration.test_proposals import ALICE, TODAY

HEADPHONES = "I returned my headphones last week, but I still haven't received my refund."
SPECIALIST = AgentConfig("fake", "fake-model", "v1", "specialist")

AGENTS = {
    "You classify": "classifier",
    "You are the investigation agent": "investigation",
    "You are the policy agent": "policy",
    "You are the resolution agent": "resolution",
    "You are a customer support agent": "single",
}


def agent_of(call: FakeCall) -> str:
    return next(name for prefix, name in AGENTS.items() if call.system.startswith(prefix))


Plan = list[list[tuple[str, dict[str, Any]]]]


@dataclass
class Scenario:
    """Per agent: tool-call turns, then the JSON it returns."""

    outputs: dict[str, dict[str, Any] | str]
    plans: dict[str, Plan | Callable[[FakeCall], Plan]] = field(default_factory=dict)

    def __call__(self, call: FakeCall):
        agent = agent_of(call)
        if call.response_schema is not None:
            output = self.outputs[agent]
            return reply(output) if isinstance(output, str) else json_reply(output)
        plan = self.plans.get(agent, [])
        plan = plan(call) if callable(plan) else plan
        turns = sum(1 for m in call.messages if m.role == "assistant")
        return reply(tool_calls=plan[turns]) if turns < len(plan) else reply("I have the facts.")


# --- Fixtures --------------------------------------------------------------------------


@pytest.fixture
def db(conn: Connection) -> Iterator[Session]:
    seed(conn, TODAY)
    ingest(conn, FakeEmbedder(), load_policies())
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
def make_services(providers) -> Callable[..., WorkflowServices]:
    payments, shipping = providers
    saver = InMemorySaver()

    def make(script, config: AgentConfig = SPECIALIST) -> WorkflowServices:
        return WorkflowServices(
            provider=ScriptedModel(script),
            config=config,
            payments=payments,
            shipping=shipping,
            embedder=FakeEmbedder(),
            checkpointer=saver,
            today=TODAY,
            sleep=lambda seconds: None,
        )

    return make


def open_ticket(db: Session, message: str, order_id: str | None) -> str:
    return create_ticket(db, ALICE, TicketCreate(message=message, order_id=order_id)).id


def event_types(db: Session, run_id: str) -> list[str]:
    return list(
        db.execute(
            text("SELECT event_type FROM support.audit_events WHERE run_id = :r ORDER BY id"),
            {"r": run_id},
        ).scalars()
    )


def refund_ids(db: Session, payments: ProviderClient) -> tuple[str, str]:
    return_id = db.execute(
        text("SELECT id FROM support.returns WHERE order_id = 'ord_1001'")
    ).scalar_one()
    return return_id, payments.get("/payments/ord_1001")["payment_id"]


def headphones_scenario(return_id: str, payment_id: str, **proposal: Any) -> Scenario:
    return Scenario(
        outputs={
            "classifier": {
                "category": "refund_request",
                "order_id": "ord_1001",
                "summary": "Refund for returned headphones not yet received.",
            },
            "investigation": {
                "order_id": "ord_1001",
                "summary": "Return received, inspection passed, no refunds yet.",
                "evidence_refs": ["ord_1001", return_id, payment_id],
                "missing_information": [],
            },
            "policy": {
                "interpretation": "Eligible once received and inspected; reviewer approves.",
                "policy_refs": ["refund_policy_v1:section_3"],
                "needs_human_review": False,
            },
            "resolution": {
                "action": "request_refund_approval",
                "order_id": "ord_1001",
                "amount_minor": 249900,
                "currency": "INR",
                "reason": "Return received and inspection passed",
                "evidence_refs": [return_id, payment_id],
                "policy_refs": ["refund_policy_v1:section_3"],
                "missing_information": [],
                **proposal,
            },
        },
        plans={
            "investigation": [
                [
                    ("get_order", {"order_id": "ord_1001"}),
                    ("get_return_status", {"order_id": "ord_1001"}),
                    ("get_payment_status", {"order_id": "ord_1001"}),
                ]
            ],
            "policy": [
                [
                    (
                        "search_policy",
                        {
                            "query": REFUND_QUESTION,
                            "policy_keys": ["refund_policy"],
                            "effective_on": "2026-10-17",
                        },
                    )
                ]
            ],
        },
    )


# --- Week 2 gate: the headphones refund ------------------------------------------------


def test_headphones_refund_produces_a_validated_proposal_with_citations(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    return_id, payment_id = refund_ids(db, payments)
    services = make_services(headphones_scenario(return_id, payment_id))
    ticket_id = open_ticket(db, HEADPHONES, "ord_1001")

    run = start_resolution(db, ALICE, ticket_id, services)

    assert run.status == "waiting_for_approval"
    assert run.output["route"] == "request_approval"
    assert run.output["policy_references"][0] == "refund_policy_v1:section_3"
    assert run.output["proposal_id"].startswith("prp_")
    assert "No refund has been issued yet" in run.output["final_response"]
    proposal = db.execute(
        text("SELECT payload FROM support.action_proposals WHERE id = :p"),
        {"p": run.output["proposal_id"]},
    ).scalar_one()
    assert proposal["amount_minor"] == 249900
    assert proposal["evidence_refs"] == [return_id, payment_id]
    # Proposed, not paid.
    assert payments.get("/payments/ord_1001")["refunds"] == []
    ticket = get_ticket(db, ALICE, ticket_id)
    assert (ticket.status, ticket.category) == ("awaiting_approval", "refund_request")

    events = event_types(db, run.id)
    assert events[0:2] == ["run_started", "run_configured"]
    assert events.count("model_call") == len(services.provider.calls)
    assert {"tool_requested", "tool_finished", "proposal_validated", "proposal_created"} <= set(
        events
    )
    models = (
        db.execute(
            text(
                "SELECT redacted_payload FROM support.audit_events WHERE run_id = :r "
                "AND event_type = 'model_call'"
            ),
            {"r": run.id},
        )
        .scalars()
        .all()
    )
    assert all(m["input_tokens"] == 100 and m["prompt"].endswith("_v1") for m in models)


def test_single_agent_baseline_reaches_the_same_proposal(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    return_id, payment_id = refund_ids(db, payments)
    specialist = headphones_scenario(return_id, payment_id)
    single = Scenario(
        outputs={
            "classifier": specialist.outputs["classifier"],
            "single": specialist.outputs["resolution"],
        },
        plans={"single": specialist.plans["investigation"] + specialist.plans["policy"]},
    )
    services = make_services(single, replace(SPECIALIST, architecture="single"))

    run = start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)

    assert run.status == "waiting_for_approval"
    assert {agent_of(c) for c in services.provider.calls} == {"classifier", "single"}


def test_policy_version_is_chosen_after_facts_for_refunds(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    services = make_services(headphones_scenario(*refund_ids(db, payments)))
    start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)

    order = [agent_of(c) for c in services.provider.calls]
    assert order.index("policy") > max(i for i, a in enumerate(order) if a == "investigation")
    policy_brief = next(c for c in services.provider.calls if agent_of(c) == "policy")
    assert "Verified facts" in policy_brief.messages[0].text


# --- Other routes ----------------------------------------------------------------------


def test_order_status_is_answered_and_the_ticket_resolved(db: Session, make_services) -> None:
    scenario = Scenario(
        outputs={
            "classifier": {
                "category": "order_status",
                "order_id": "ord_1008",
                "summary": "Customer asks where their order is.",
            },
            "investigation": {
                "order_id": "ord_1008",
                "summary": "In transit.",
                "evidence_refs": ["ord_1008"],
                "missing_information": [],
            },
            "policy": {
                "interpretation": "Share the latest courier status.",
                "policy_refs": [],
                "needs_human_review": False,
            },
            "resolution": {
                "action": "answer_only",
                "order_id": "ord_1008",
                "reason": "Status request",
                "evidence_refs": ["ord_1008"],
                "customer_message": "Your order ord_1008 is on its way.",
            },
        },
        plans={
            "investigation": [
                [
                    ("get_order", {"order_id": "ord_1008"}),
                    ("get_shipment_status", {"order_id": "ord_1008"}),
                ]
            ],
            "policy": [[("search_policy", {"query": "shipment status delayed order"})]],
        },
    )
    services = make_services(scenario)
    ticket_id = open_ticket(db, "Where is my order?", "ord_1008")

    run = start_resolution(db, ALICE, ticket_id, services)

    assert run.status == "completed"
    assert run.output["final_response"] == "Your order ord_1008 is on its way."
    assert get_ticket(db, ALICE, ticket_id).status == "resolved"


def test_out_of_scope_complaint_is_escalated_without_other_agents(
    db: Session, make_services
) -> None:
    scenario = Scenario(
        outputs={
            "classifier": {
                "category": "out_of_scope",
                "order_id": None,
                "summary": "Asks to change account email.",
            }
        }
    )
    services = make_services(scenario)
    ticket_id = open_ticket(db, "Please change my account email.", None)

    run = start_resolution(db, ALICE, ticket_id, services)

    assert run.status == "escalated"
    assert [agent_of(c) for c in services.provider.calls] == ["classifier"]
    assert get_ticket(db, ALICE, ticket_id).status == "escalated"


def test_missing_order_is_asked_for_and_the_reply_resumes_the_run(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    base = headphones_scenario(*refund_ids(db, payments))

    def investigation_plan(call: FakeCall) -> Plan:
        if "ord_1001" in call.messages[0].text:
            return base.plans["investigation"]  # type: ignore[return-value]
        return [[("list_orders", {})]]

    scenario = replace(
        base,
        outputs={**base.outputs, "classifier": {**base.outputs["classifier"], "order_id": None}},
        plans={**base.plans, "investigation": investigation_plan},
    )
    services = make_services(scenario)
    ticket_id = open_ticket(db, "Where is my refund?", None)

    paused = start_resolution(db, ALICE, ticket_id, services)

    assert paused.status == "waiting_for_customer"
    assert "ord_1001" in paused.output["final_response"]  # recent orders are listed
    assert get_ticket(db, ALICE, ticket_id).status == "awaiting_customer"

    resumed = resume_with_message(db, ALICE, paused.id, "It's order ord_1001.", services)

    assert resumed.status == "waiting_for_approval"
    assert resumed.output["order_id"] == "ord_1001"
    assert "customer_message" in event_types(db, paused.id)


def test_pause_and_resume_survive_in_the_postgres_checkpointer(
    db: Session, make_services, providers, database_url: str
) -> None:
    payments, _ = providers
    base = headphones_scenario(*refund_ids(db, payments))
    scenario = replace(
        base,
        outputs={**base.outputs, "classifier": {**base.outputs["classifier"], "order_id": None}},
        plans={
            **base.plans,
            "investigation": lambda call: (
                base.plans["investigation"]
                if "ord_1001" in call.messages[0].text
                else [[("list_orders", {})]]
            ),
        },
    )
    saver = postgres_checkpointer(database_url)
    paused = start_resolution(
        db,
        ALICE,
        open_ticket(db, "Where is my refund?", None),
        replace(make_services(scenario), checkpointer=saver),
    )
    assert paused.status == "waiting_for_customer"

    # A new saver instance, as after an application restart: state comes from PostgreSQL.
    restarted = replace(make_services(scenario), checkpointer=postgres_checkpointer(database_url))
    resumed = resume_with_message(db, ALICE, paused.id, "ord_1001", restarted)

    assert resumed.status == "waiting_for_approval"
    tables = db.execute(
        text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'langgraph'")
    ).scalar_one()
    assert tables > 0


def test_reply_to_a_run_that_is_not_waiting_is_refused(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    services = make_services(headphones_scenario(*refund_ids(db, payments)))
    run = start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)

    with pytest.raises(ServiceFailure) as raised:
        resume_with_message(db, ALICE, run.id, "hello", services)
    assert raised.value.error_code == "RUN_NOT_WAITING"


# --- Guardrails (W2.12, W2.15) ---------------------------------------------------------


def test_invented_evidence_ref_is_escalated(db: Session, make_services, providers) -> None:
    payments, _ = providers
    return_id, payment_id = refund_ids(db, payments)
    services = make_services(
        headphones_scenario(return_id, payment_id, evidence_refs=["ret_made_up"])
    )

    run = start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)

    assert run.status == "escalated"
    assert run.output["validation_problems"] == ["UNKNOWN_EVIDENCE_REF:ret_made_up"]
    assert run.output["proposal_id"] is None


def test_amount_above_the_allowed_refund_is_escalated(
    db: Session, make_services, providers
) -> None:
    payments, _ = providers
    services = make_services(headphones_scenario(*refund_ids(db, payments), amount_minor=999900))

    run = start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)

    assert run.status == "escalated"
    assert run.output["validation_problems"][0].startswith("NOT_ELIGIBLE")


def test_invalid_json_gets_one_repair_then_escalates(db: Session, make_services, providers) -> None:
    payments, _ = providers
    scenario = headphones_scenario(*refund_ids(db, payments))
    scenario.outputs["resolution"] = "this is not json"
    services = make_services(scenario)

    run = start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)

    assert run.status == "escalated"
    assert run.output["stop_reason"] == "invalid_model_output"
    assert event_types(db, run.id).count("model_output_invalid") == 2


def test_step_limit_stops_the_run(db: Session, make_services, providers) -> None:
    payments, _ = providers
    services = make_services(
        headphones_scenario(*refund_ids(db, payments)), replace(SPECIALIST, max_steps=3)
    )

    run = start_resolution(db, ALICE, open_ticket(db, HEADPHONES, "ord_1001"), services)

    assert run.status == "escalated"
    assert run.output["stop_reason"] == "max_steps"
    assert "run_halted" in event_types(db, run.id)


def test_another_customers_order_is_never_used(db: Session, make_services) -> None:
    scenario = Scenario(
        outputs={
            "classifier": {
                "category": "refund_request",
                "order_id": "ord_1003",
                "summary": "Refund request.",
            },
            "investigation": {
                "order_id": None,
                "summary": "Order not found.",
                "evidence_refs": [],
                "missing_information": ["order_id"],
            },
            "policy": {"interpretation": "", "policy_refs": [], "needs_human_review": False},
        },
        plans={
            "investigation": [[("get_order", {"order_id": "ord_1003"})]],
            "policy": [[("search_policy", {"query": REFUND_QUESTION})]],
        },
    )
    services = make_services(scenario)

    run = start_resolution(db, ALICE, open_ticket(db, "Refund ord_1003 please", None), services)

    assert run.status == "waiting_for_customer"
    assert "tool_rejected" in event_types(db, run.id)
    assert "ord_1003" not in run.output["final_response"]
