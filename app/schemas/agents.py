"""Validated output schemas for the model's structured answers (W2.7, W2.8).

The model fills these as JSON; Pydantic validates them, and the business-rule validator
(app/workflows/validation.py) then checks every value independently: identity, ownership,
eligibility and amounts are never taken from these objects on trust.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["order_status", "refund_request", "damaged_item", "out_of_scope"]
Action = Literal["answer_only", "request_refund_approval", "escalate", "ask_clarification"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Classification(StrictModel):
    """W2.8: one of the three supported complaint types, or out of scope."""

    category: Category
    # An order ID the customer wrote, if any (e.g. ord_1042). Ownership is checked later.
    order_id: str | None = None
    # One neutral sentence for the ticket summary; no personal data beyond the order.
    summary: str = Field(max_length=300)


class EvidenceSummary(StrictModel):
    """Investigation agent output: what the read tools showed."""

    order_id: str | None = None
    summary: str = Field(max_length=1500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    missing_information: list[str] = Field(default_factory=list, max_length=5)


class PolicyFinding(StrictModel):
    """Policy agent output: which passages apply and what they mean for this case."""

    interpretation: str = Field(max_length=1500)
    policy_refs: list[str] = Field(default_factory=list, max_length=10)
    # True when no policy applies or the evidence conflicts (W2.2): a human decides.
    needs_human_review: bool = False


class ResolutionProposal(StrictModel):
    """W2.7: the resolution proposal, e.g.

    {"action": "request_refund_approval", "order_id": "ord_1042", "amount_minor": 249900,
     "currency": "INR", "reason": "Return received and inspection passed",
     "evidence_refs": ["ret_42", "pay_92"], "policy_refs": ["refund_policy_v1:section_3"],
     "missing_information": []}
    """

    action: Action
    order_id: str | None = None
    amount_minor: int | None = Field(default=None, ge=0)
    currency: str | None = None
    reason: str = Field(max_length=500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)
    policy_refs: list[str] = Field(default_factory=list, max_length=10)
    missing_information: list[str] = Field(default_factory=list, max_length=5)
    # For answer_only and ask_clarification: the message to the customer. For refunds and
    # escalations the workflow writes the message itself from the verified outcome.
    customer_message: str = Field(default="", max_length=1500)
