"""Structured workflow state (W2.6): enough to inspect a run and resume it from a checkpoint.

Values are plain JSON-compatible data, so the LangGraph checkpointer can store them. Facts and
citations are written by code from successful tool results, never copied from model text.
"""

import operator
from typing import Annotated, Any, TypedDict


def merge(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Parallel agents (W2.10) each add their own keys to the same dict."""
    return {**(left or {}), **(right or {})}


class WorkflowState(TypedDict, total=False):
    # Identity and scope, set by code from the authenticated request.
    run_id: str
    ticket_id: str
    authenticated_customer_id: str
    order_id: str | None
    complaint_category: str | None
    customer_message: str
    # Every message from the customer, oldest first (clarification answers are appended).
    customer_messages: Annotated[list[str], operator.add]
    summary: str

    # Evidence, keyed by tool and argument (e.g. "get_order:ord_1001"), from tool results.
    verified_facts: Annotated[dict[str, Any], merge]
    # Citations returned by search_policy in this run.
    policy_references: Annotated[list[str], operator.add]
    # Specialist outputs (validated schemas, as dicts).
    evidence: dict[str, Any] | None
    policy_finding: dict[str, Any] | None
    proposal: dict[str, Any] | None
    missing_information: list[str]
    clarification_rounds: int

    # Outcome.
    # answer_only | request_approval | escalate | ask_clarification
    # | refunded | refund_unconfirmed | rejected
    route: str | None
    validation_problems: list[str]
    proposed_action_id: str | None
    approval_id: str | None
    # The stored reviewer decision, read after resuming (W3.3): approved | rejected | expired
    approval_decision: str | None
    # How many approvals this run asked for, and what the last revalidation found (W3.6).
    approval_rounds: int
    revalidation: str | None  # unchanged | changed
    refund_operation_id: str | None
    execution_status: str | None  # the refund operation's status: succeeded | submitted | ...
    external_result_reference: str | None  # provider reference
    retry_counts: dict[str, int]
    error_details: dict[str, Any] | None
    stop_reason: str | None
    final_response: str | None
