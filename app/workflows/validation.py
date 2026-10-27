"""Business-rule validation of a resolution proposal, in ordinary code (W2.11, W2.12).

The model proposes; this module decides the route. It checks the proposal against the facts
the tools returned and against the eligibility code (W1.14), never against the model's own
claims:

- every evidence ref must be a record a read tool returned in this run
- every policy ref must be a citation search_policy returned in this run
- a refund needs a verified order of this customer, the store currency, evidence, a refund
  policy citation, and an `eligible` decision for exactly that amount

Anything that fails goes to a human (escalate), with the problems recorded.
"""

from dataclasses import dataclass, field
from typing import Any

from app.agents.facts import known_refs
from app.config import get_settings
from app.schemas.agents import ResolutionProposal
from app.services.eligibility import EligibilityDecision, check_refund_eligibility
from app.services.errors import ServiceFailure
from app.tools.base import ToolContext

Route = str  # answer_only | request_approval | escalate | ask_clarification


@dataclass
class Validation:
    route: Route
    problems: list[str] = field(default_factory=list)
    decision: EligibilityDecision | None = None


def validate_proposal(
    ctx: ToolContext,
    proposal: ResolutionProposal,
    *,
    facts: dict[str, Any],
    citations: list[str],
) -> Validation:
    problems = [
        f"UNKNOWN_EVIDENCE_REF:{ref}"
        for ref in proposal.evidence_refs
        if ref not in known_refs(facts)
    ] + [f"UNKNOWN_POLICY_REF:{ref}" for ref in proposal.policy_refs if ref not in citations]
    if problems:
        return Validation("escalate", problems)

    if proposal.action == "escalate":
        return Validation("escalate")
    if proposal.action == "ask_clarification":
        if not proposal.missing_information or not proposal.customer_message.strip():
            return Validation("escalate", ["CLARIFICATION_WITHOUT_QUESTION"])
        return Validation("ask_clarification")
    if proposal.action == "answer_only":
        if not proposal.customer_message.strip():
            return Validation("escalate", ["ANSWER_WITHOUT_MESSAGE"])
        return Validation("answer_only")
    return validate_refund(ctx, proposal, facts)


def validate_refund(
    ctx: ToolContext, proposal: ResolutionProposal, facts: dict[str, Any]
) -> Validation:
    problems: list[str] = []
    # The proposal's order must be one get_order returned in this run (so it is the customer's).
    if not proposal.order_id or f"get_order:{proposal.order_id}" not in facts:
        problems.append("ORDER_NOT_VERIFIED")
    if proposal.currency != get_settings().store_currency:
        problems.append("WRONG_CURRENCY")
    if not proposal.amount_minor:
        problems.append("AMOUNT_MISSING")
    if not proposal.evidence_refs:
        problems.append("EVIDENCE_MISSING")
    if not any(ref.startswith("refund_policy_") for ref in proposal.policy_refs):
        problems.append("REFUND_POLICY_NOT_CITED")
    if problems:
        return Validation("escalate", problems)

    try:
        decision = check_refund_eligibility(
            ctx.db,
            ctx.payments,
            ctx.customer,
            proposal.order_id,  # type: ignore[arg-type]
            amount_minor=proposal.amount_minor,
            today=ctx.today,
        )
    except ServiceFailure as failure:
        # No decision without the payment provider's view (W1.14); a person takes over.
        return Validation("escalate", [f"ELIGIBILITY_UNAVAILABLE:{failure.error_code}"])
    if decision.outcome != "eligible":
        return Validation(
            "escalate", [f"NOT_ELIGIBLE:{decision.outcome}:{decision.reason}"], decision
        )
    if decision.amount_minor != proposal.amount_minor:
        return Validation("escalate", ["AMOUNT_NOT_ALLOWED"], decision)
    return Validation("request_approval", [], decision)
