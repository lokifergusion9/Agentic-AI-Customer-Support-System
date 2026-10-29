"""Evaluation task format: one YAML file per task, split into two views (W4.1, E§8, §15).

The agent view is only `agent_input`: what a real customer would send. Everything else (the
fixture, the reviewer behaviour and the grading rules) is the evaluator view and never reaches
the agent (E§5). Every model forbids unknown keys, so a typo in a task file fails loudly instead
of silently dropping an expectation.

Money is in paise, like the `support` schema. Dates in a fixture are absolute; `as_of` is the
day the environment clock is set to when the task runs (W4.4).
"""

import re
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.registry import TOOLS

TaskId = str
Category = Literal[
    "read_only_lookup",
    "eligible_refund",
    "ineligible_refund",
    "damaged_item",
    "missing_information",
    "access_boundary",
    "tool_failure",
    "duplicate_operation",
    "untrusted_content",
]
TicketStatus = Literal["open", "awaiting_customer", "awaiting_approval", "escalated", "resolved"]
RunStatus = Literal[
    "running", "waiting_for_customer", "waiting_for_approval", "completed", "escalated", "failed"
]

# Steps a required_behavior or forbidden_actions entry may name: every agent tool, plus the
# approval milestones the reviewer simulator produces (W4.5).
APPROVAL_STEPS = frozenset(
    {"approval_requested", "approval_granted", "approval_rejected", "approval_expired"}
)
KNOWN_STEPS = frozenset(TOOLS) | APPROVAL_STEPS

POLICY_REF = re.compile(r"[a-z_]+_v\d+:section_\d+")

# Invariants the evaluators check on every attempt where they apply (W4.9).
Invariant = Literal[
    "refund_requires_valid_approval",
    "no_cross_customer_access",
    "no_duplicate_refund_effect",
    "refund_within_refundable_balance",
    "no_side_effect_without_verification",
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- agent view -----------------------------------------------------------------------------


class AgentInput(Strict):
    customer_subject: str = Field(min_length=1)  # becomes the JWT sub, never a tool argument
    order_id: str | None = None  # as the customer would give it, if they did
    user_message: str = Field(min_length=1)


# --- environment fixture ----------------------------------------------------------------------


class CustomerFixture(Strict):
    id: str
    external_auth_subject: str
    name: str
    email: str


class OrderItemFixture(Strict):
    product_id: str
    name: str
    amount_minor: int = Field(gt=0)
    quantity: int = Field(default=1, gt=0)


class OrderFixture(Strict):
    id: str
    customer_id: str
    status: Literal["placed", "shipped", "delivered", "cancelled"]
    placed_at: date
    delivered_at: date | None = None
    shipping_minor: int = Field(default=0, ge=0)
    items: list[OrderItemFixture] = Field(min_length=1)

    @property
    def paid_amount_minor(self) -> int:
        return sum(i.amount_minor * i.quantity for i in self.items) + self.shipping_minor

    @model_validator(mode="after")
    def _delivered_date(self) -> Self:
        if (self.status == "delivered") != (self.delivered_at is not None):
            raise ValueError("delivered_at is set exactly when status is delivered")
        return self


class ReturnFixture(Strict):
    order_id: str
    status: Literal["requested", "in_transit", "received", "closed"]
    requested_at: date
    received_at: date | None = None
    inspection_status: Literal["pending", "passed", "failed"] | None = None

    @model_validator(mode="after")
    def _received(self) -> Self:
        # The same rules as the returns table checks (migration 0005).
        if (self.status in ("received", "closed")) != (self.received_at is not None):
            raise ValueError("received_at is set exactly when the return was received")
        if (self.inspection_status is None) != (self.received_at is None):
            raise ValueError("only a received return has an inspection_status")
        return self


class PastRefundFixture(Strict):
    status: Literal["submitted", "succeeded"]
    amount_minor: int = Field(gt=0)


class PaymentFixture(Strict):
    # Pushed to the mock payment API, which is the source of truth for payments (Q2).
    order_id: str
    amount_minor: int = Field(gt=0)
    refunds: list[PastRefundFixture] = []


class ShipmentFixture(Strict):
    order_id: str
    status: Literal["label_created", "in_transit", "out_for_delivery", "delivered", "exception"]
    estimated_delivery: date


class FaultFixture(Strict):
    # Keyed by logical operation, not by HTTP call, so a retry does not skip the fault (W4.4).
    operation: str
    kind: Literal["timeout", "http_5xx", "unknown_outcome"]
    on_call: int = Field(default=1, gt=0)  # which call of that operation fails
    times: int = Field(default=1, gt=0)  # how many consecutive calls fail


class EnvironmentFixture(Strict):
    as_of: date
    customers: list[CustomerFixture] = Field(min_length=1)
    orders: list[OrderFixture] = []
    returns: list[ReturnFixture] = []
    payments: list[PaymentFixture] = []
    shipments: list[ShipmentFixture] = []
    faults: list[FaultFixture] = []

    @model_validator(mode="after")
    def _references(self) -> Self:
        customer_ids = {c.id for c in self.customers}
        order_ids = {o.id for o in self.orders}
        if unknown := {o.customer_id for o in self.orders} - customer_ids:
            raise ValueError(f"orders reference unknown customers: {sorted(unknown)}")
        for kind, rows in (
            ("returns", self.returns),
            ("payments", self.payments),
            ("shipments", self.shipments),
        ):
            if unknown := {r.order_id for r in rows} - order_ids:
                raise ValueError(f"{kind} reference unknown orders: {sorted(unknown)}")
        return self


# --- reviewer behaviour -------------------------------------------------------------------------


class PermittedAction(Strict):
    action: Literal["refund"]
    order_id: str
    amount_minor: int = Field(gt=0)


class ReviewerBehavior(Strict):
    # The simulator approves only a proposal matching applies_to; anything else is rejected (W4.5).
    decision: Literal["approve", "reject", "expire"]
    applies_to: PermittedAction


# --- evaluation --------------------------------------------------------------------------------


class ExpectedOutcome(Strict):
    """Expected final business state, read from the database and the mocks, not the message.

    refund_total_minor and refund_count cover refunds made during the attempt; refunds already
    in the fixture (`payments[].refunds`) are not counted.
    """

    refund_total_minor: int = Field(default=0, ge=0)
    refund_count: int = Field(default=0, ge=0)
    ticket_status: TicketStatus
    run_status: RunStatus
    approval_requested: bool = False


class OrderedSteps(Strict):
    """A partial-order constraint (E§13): every step in `requires` happens before `before`."""

    before: str
    requires: list[str] = Field(min_length=1)


class ResponseChecks(Strict):
    must_mention: list[str] = []
    must_not_mention: list[str] = []  # e.g. details of another customer's order
    must_not_claim: list[str] = []  # e.g. "refund has been issued" when it was not


class Limits(Strict):
    max_model_turns: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    timeout_seconds: int = Field(gt=0)


class Evaluation(Strict):
    # The original §15 expected-outcome list: relevant customer and order, applicable policy,
    # allowed and forbidden actions, whether approval is required, expected final business
    # state, and expected escalation or clarification.
    expected_customer_id: str
    expected_order_id: str | None = None
    # Citations as search_policy returns them: "<document>_v<n>:section_<n>".
    applicable_policy: list[str] = []
    resolution: Literal["resolve", "refuse", "escalate", "clarify"]
    approval_required: bool
    allowed_actions: list[str] = []
    forbidden_actions: list[str] = []
    outcome: ExpectedOutcome
    invariants: list[Invariant] = []
    required_behavior: list[OrderedSteps] = []
    response_checks: ResponseChecks = ResponseChecks()
    limits: Limits

    @model_validator(mode="after")
    def _steps_and_policy(self) -> Self:
        named = set(self.allowed_actions) | set(self.forbidden_actions)
        for rule in self.required_behavior:
            named |= {rule.before, *rule.requires}
        if unknown := named - KNOWN_STEPS:
            raise ValueError(f"unknown steps: {sorted(unknown)}")
        if overlap := set(self.allowed_actions) & set(self.forbidden_actions):
            raise ValueError(f"steps both allowed and forbidden: {sorted(overlap)}")
        for ref in self.applicable_policy:
            if not POLICY_REF.fullmatch(ref):
                raise ValueError(f"policy reference {ref!r} is not <document>_v<n>:section_<n>")
        if self.approval_required != self.outcome.approval_requested:
            raise ValueError("approval_required and outcome.approval_requested disagree")
        return self


# --- the task ----------------------------------------------------------------------------------


class Task(Strict):
    id: TaskId = Field(pattern=r"^[a-z0-9_]+$")
    version: int = Field(gt=0)
    category: Category
    difficulty: Literal["easy", "medium", "hard"]
    scenario_family: str = Field(pattern=r"^[a-z0-9_]+$")
    agent_input: AgentInput
    environment_fixture: EnvironmentFixture
    reviewer_behavior: ReviewerBehavior | None = None
    evaluation: Evaluation

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        fixture, evaluation = self.environment_fixture, self.evaluation
        customer = next(
            (
                c
                for c in fixture.customers
                if c.external_auth_subject == self.agent_input.customer_subject
            ),
            None,
        )
        if customer is None:
            raise ValueError("agent_input.customer_subject is not a fixture customer")
        if evaluation.expected_customer_id != customer.id:
            raise ValueError("expected_customer_id is not the authenticated customer")
        if evaluation.approval_required != (self.reviewer_behavior is not None):
            raise ValueError("reviewer_behavior is given exactly when approval is required")
        if self.category == "missing_information" and evaluation.resolution != "clarify":
            raise ValueError("missing_information tasks expect a clarification")
        if evaluation.resolution == "clarify":
            # Single-turn clarification: the run stops for the customer with no side effects.
            outcome = evaluation.outcome
            if (outcome.ticket_status, outcome.run_status) != (
                "awaiting_customer",
                "waiting_for_customer",
            ):
                raise ValueError("a clarification task stops in awaiting_customer")
            if outcome.refund_count or outcome.refund_total_minor or outcome.approval_requested:
                raise ValueError("a clarification task has no side effects")
        return self


class AgentView(Strict):
    """All the adapter may hand to the agent."""

    task_id: TaskId
    agent_input: AgentInput


class EvaluatorView(Strict):
    """Everything else: hidden from the agent, read by the environment, simulator and evaluators."""

    task_id: TaskId
    version: int
    content_hash: str
    category: Category
    difficulty: Literal["easy", "medium", "hard"]
    scenario_family: str
    environment_fixture: EnvironmentFixture
    reviewer_behavior: ReviewerBehavior | None
    evaluation: Evaluation
