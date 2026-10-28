"""The resolution workflow as a LangGraph graph (W2.9–W2.13, HLD §2).

    classify ─┬─ out of scope ────────────────────────────────────────────► finish_escalate
              ├─ single architecture ─► single_agent ─┐
              ├─ refund ─► investigate_then_policy ───┤  (policy version depends on facts)
              └─ other ─► investigate ∥ policy ───────┤  (independent, run concurrently)
                                                      ▼
                       check_missing ─ missing ─► request_clarification ─► await_customer
                             │                                  (interrupt; resumes at classify)
                             ▼
                 resolve (specialists) ─► validate (code) ─┬─► persist_proposal ─► finish_approval
                                                           ├─► finish_answer               │
                                                           ├─► finish_escalate             ▼
                                                           └─► request_clarification  await_approval
                                         (interrupt; resumed after the reviewer decides, W3.3) │
                                                                                               ▼
                 check_decision (stored approval) ─┬─ approved | expired ─► revalidate
                                                   ├─ rejected ─► finish_rejected
                                                   └─ undecided ─► await_approval

                 revalidate (eligibility + payload hash, W3.6)
                   ├─ unchanged, approved ─► execute_refund ─► verify_refund (below)
                   ├─ unchanged, expired ──► renew_approval ─► finish_approval (waits again)
                   ├─ changed, eligible ───► new proposal ───► finish_approval (waits again)
                   └─ no longer eligible ──► finish_escalate

                 verify_refund (asks the provider by key, W3.11)
                   ├─ succeeded ─► finish_refunded
                   ├─ failed ────► finish_refund_failed
                   └─ not yet ───► report_pending ─► await_provider (interrupt; resumed by the
                                                     reconciliation sweep, W3.12) ─► verify_refund

Model nodes propose; every route is decided by code. Side effects never happen in the same node
as `interrupt()`, because an interrupted node restarts from the top when it resumes (W3.3). The
refund is executed in its own node after the pause, and only from the stored approval: the value
the run is resumed with is a wake-up call, never an authorization (W3.7).
"""

import re
from collections.abc import Callable
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agents.facts import collect_citations, collect_facts, verified_order
from app.agents.loop import AgentRuntime, InvalidModelOutput
from app.agents.specialists import classify, find_policy, investigate, resolve, single_agent
from app.llm.calls import BudgetExceeded
from app.llm.types import ModelError
from app.schemas.agents import ResolutionProposal
from app.services.approvals import get_approval
from app.services.eligibility import check_refund_eligibility
from app.services.errors import ServiceFailure
from app.services.orders import list_orders
from app.services.proposals import (
    create_refund_proposal,
    get_proposal,
    invalidate_proposal,
    payload_hash,
    refund_payload,
    renew_approval,
)
from app.services.refunds import FINAL, confirm_refund
from app.services.runs import set_run_output, set_run_status
from app.tools.gateway import ToolGateway
from app.workflows.state import WorkflowState
from app.workflows.validation import validate_proposal

MAX_CLARIFICATION_ROUNDS = 2
# The trace's stop reasons (W3.15): budget limits keep their name, every other halt is an error.
AGENT_STOP = {"max_steps": "max_steps", "token_limit": "token_limit"}
# Approvals asked for in one run (the first, plus renewals and re-proposals, W3.6).
MAX_APPROVAL_ROUNDS = 3
# Provider checks in one go before the run waits for the reconciliation sweep (W3.11).
VERIFY_ATTEMPTS = 3
VERIFY_BACKOFF_SECONDS = 1.0
ORDER_ID = re.compile(r"\bord_\d+\b")
WORKFLOW = "system:workflow"


def highest(left: int | None, right: int | None) -> int:
    return max(left or 0, right or 0)


class GraphState(WorkflowState, total=False):
    # Budget counters survive pauses; parallel nodes report the same shared counter.
    steps_used: Annotated[int, highest]
    tokens_used: Annotated[int, highest]
    halted: bool


def build_graph(rt: AgentRuntime) -> StateGraph:
    ctx = rt.ctx

    def usage() -> dict[str, int]:
        return {"steps_used": rt.budget.steps_used, "tokens_used": rt.budget.tokens_used}

    def guarded(node: Callable[[GraphState], dict[str, Any]]):
        """Stop the run cleanly on the W2.15 limits instead of failing it (HLD 5b)."""

        def run(state: GraphState) -> dict[str, Any]:
            try:
                update = node(state)
            except BudgetExceeded as exc:
                update = halt(exc.limit, {"message": str(exc)})
            except InvalidModelOutput as exc:
                update = halt(
                    "invalid_model_output", {"schema": exc.schema, "problems": exc.problems}
                )
            except ModelError as exc:
                update = halt("model_error", {"code": exc.code, "retryable": exc.retryable})
            return {**update, **usage()}

        run.__name__ = node.__name__
        return run

    def halt(reason: str, details: dict[str, Any]) -> dict[str, Any]:
        rt.recorder.record(WORKFLOW, "run_halted", {"stop_reason": reason, **details})
        return {
            "halted": True,
            "stop_reason": reason,
            "error_details": details,
            "route": "escalate",
        }

    def gateway(actor: str, allowed: set[str]) -> ToolGateway:
        return ToolGateway(ctx, frozenset(allowed), actor=actor)

    # --- Model nodes -------------------------------------------------------------------

    def classify_node(state: GraphState) -> dict[str, Any]:
        result = classify(rt, state)
        # An order ID in the newest message wins (it may answer a clarification question),
        # then the ticket's order, then one the classifier saw.
        messages = state.get("customer_messages") or [state["customer_message"]]
        written = ORDER_ID.findall(messages[-1])
        order_id = written[-1] if written else (state.get("order_id") or result.order_id)
        rt.recorder.record(
            "agent:classifier", "complaint_classified", {"category": result.category}
        )
        return {
            "complaint_category": result.category,
            "summary": result.summary,
            "order_id": order_id,
            # A new classification starts a fresh proposal (e.g. after a clarification).
            "proposal": None,
            "evidence": None,
            "policy_finding": None,
            "missing_information": [],
            "route": None,
        }

    def single_agent_node(state: GraphState) -> dict[str, Any]:
        out = single_agent(rt, state)
        return {
            "verified_facts": collect_facts(out.tool_records),
            "policy_references": collect_citations(out.tool_records),
            "proposal": out.output.model_dump(),
        }

    def investigate_node(state: GraphState) -> dict[str, Any]:
        out = investigate(rt, state)
        return {
            "verified_facts": collect_facts(out.tool_records),
            "evidence": out.output.model_dump(),
        }

    def policy_node(state: GraphState) -> dict[str, Any]:
        out = find_policy(rt, state)
        return {
            "policy_references": collect_citations(out.tool_records),
            "policy_finding": out.output.model_dump(),
        }

    def investigate_then_policy_node(state: GraphState) -> dict[str, Any]:
        found = investigate(rt, state)
        facts = {**(state.get("verified_facts") or {}), **collect_facts(found.tool_records)}
        policy = find_policy(rt, state, facts=facts)
        return {
            "verified_facts": collect_facts(found.tool_records),
            "evidence": found.output.model_dump(),
            "policy_references": collect_citations(policy.tool_records),
            "policy_finding": policy.output.model_dump(),
        }

    def resolve_node(state: GraphState) -> dict[str, Any]:
        return {"proposal": resolve(rt, state).model_dump()}

    # --- Code nodes --------------------------------------------------------------------

    def check_missing_node(state: GraphState) -> dict[str, Any]:
        """W2.13: every supported complaint needs one verified order of this customer."""
        missing: list[str] = []
        facts = state.get("verified_facts") or {}
        order = verified_order(facts, state.get("order_id"))
        if order is None:
            missing.append("order_id")
        evidence = state.get("evidence") or {}
        missing += [m for m in evidence.get("missing_information", []) if m not in missing]
        if state.get("complaint_category") == "order_status":
            # Shipment facts are needed to answer, but a missing shipment is a fact, not a
            # question for the customer.
            missing = [m for m in missing if m == "order_id"]
        update: dict[str, Any] = {"missing_information": missing}
        if order is not None:
            update["order_id"] = order["id"]
        return update

    def validate_node(state: GraphState) -> dict[str, Any]:
        proposal = ResolutionProposal.model_validate(state["proposal"])
        with ctx.lock:
            result = validate_proposal(
                ctx,
                proposal,
                facts=state.get("verified_facts") or {},
                citations=state.get("policy_references") or [],
            )
            ctx.db.rollback()  # the eligibility check only reads
        rt.recorder.record(
            WORKFLOW,
            "proposal_validated",
            {"action": proposal.action, "route": result.route, "problems": result.problems},
        )
        update: dict[str, Any] = {"route": result.route, "validation_problems": result.problems}
        if result.route == "ask_clarification":
            update["missing_information"] = proposal.missing_information
        return update

    def persist_proposal_node(state: GraphState) -> dict[str, Any]:
        proposal = state["proposal"]
        result = gateway("agent:resolution", {"create_refund_proposal"}).execute(
            "create_refund_proposal",
            {
                "order_id": proposal["order_id"],
                "amount_minor": proposal["amount_minor"],
                "reason": proposal["reason"],
                "evidence_refs": proposal["evidence_refs"],
            },
            call_id="persist_proposal",
        )
        if not result["success"]:
            return {"route": "escalate", "validation_problems": [result["error_code"]]}
        return {
            "proposed_action_id": result["data"]["proposal_id"],
            "approval_id": result["data"]["approval_id"],
        }

    def request_clarification_node(state: GraphState) -> dict[str, Any]:
        question = clarification_question(state)
        finish(
            state,
            run_status="waiting_for_customer",
            ticket_status="awaiting_customer",
            response=question,
            route="ask_clarification",
        )
        rt.recorder.record(
            WORKFLOW, "clarification_requested", {"missing": state.get("missing_information")}
        )
        return {"final_response": question, "route": "ask_clarification"}

    def await_customer_node(state: GraphState) -> dict[str, Any]:
        # Nothing else happens here: on resume this node restarts from the top (W3.3).
        answer = interrupt({"question": state.get("final_response")})
        return {
            "customer_messages": [str(answer)],
            "clarification_rounds": (state.get("clarification_rounds") or 0) + 1,
            "final_response": None,
        }

    def await_approval_node(state: GraphState) -> dict[str, Any]:
        # Nothing else happens here: on resume this node restarts from the top (W3.3).
        interrupt({"approval_id": state.get("approval_id")})
        return {}

    # --- After the reviewer's decision (code) ------------------------------------------

    def check_decision_node(state: GraphState) -> dict[str, Any]:
        # The stored decision, not the resume value, decides what happens next (W3.7).
        with ctx.lock:
            approval = get_approval(ctx.db, state["approval_id"])
        return {"approval_decision": approval.decision}

    def revalidate_node(state: GraphState) -> dict[str, Any]:
        """Recheck the approved (or expired) proposal against the facts now (W3.6, HLD §2 K).

        Unchanged means code would propose exactly the same payload today. Otherwise the old
        proposal is invalidated, and a still-eligible refund is proposed again for a new
        approval: an approval of old facts never authorizes a refund on new ones.
        """
        try:
            with ctx.lock:
                stored = get_proposal(ctx.db, state["proposed_action_id"])
                decision = check_refund_eligibility(
                    ctx.db, ctx.payments, ctx.customer, stored.order_id, today=ctx.today
                )
        except ServiceFailure as failure:
            return {"route": "escalate", "validation_problems": [failure.error_code]}
        reason = stored.payload["reason"]
        evidence = stored.payload["evidence_refs"]
        eligible = decision.outcome == "eligible"
        current = refund_payload(decision, reason=reason, evidence_refs=evidence)
        if eligible and payload_hash(current) == stored.payload_hash:
            return {"revalidation": "unchanged"}

        changed = {"order_id": stored.order_id, "outcome": decision.outcome}
        rt.recorder.record(WORKFLOW, "proposal_details_changed", changed)
        if stored.status != "invalidated":
            with ctx.lock:
                invalidate_proposal(
                    ctx.db,
                    stored.proposal_id,
                    reason=decision.reason or "details_changed",
                    actor=WORKFLOW,
                )
        rounds = state.get("approval_rounds") or 1
        if not eligible or rounds >= MAX_APPROVAL_ROUNDS:
            problem = "NOT_ELIGIBLE" if not eligible else "TOO_MANY_APPROVAL_ROUNDS"
            return {
                "revalidation": "changed",
                "route": "escalate",
                "validation_problems": [problem],
            }
        try:
            with ctx.lock:
                proposal = create_refund_proposal(
                    ctx.db,
                    ctx.customer,
                    ctx.run_id,
                    decision,
                    reason=reason,
                    evidence_refs=evidence,
                )
        except ServiceFailure as failure:
            return {"route": "escalate", "validation_problems": [failure.error_code]}
        return {
            "revalidation": "changed",
            "proposed_action_id": proposal.proposal_id,
            "approval_id": proposal.approval_id,
            "approval_decision": None,
            "approval_rounds": rounds + 1,
            "proposal": {**state["proposal"], "amount_minor": decision.amount_minor},
        }

    def renew_approval_node(state: GraphState) -> dict[str, Any]:
        # Unchanged facts, expired approval: the same proposal needs a new approval (W3.6).
        rounds = state.get("approval_rounds") or 1
        if rounds >= MAX_APPROVAL_ROUNDS:
            return {"route": "escalate", "validation_problems": ["TOO_MANY_APPROVAL_ROUNDS"]}
        with ctx.lock:
            proposal = renew_approval(ctx.db, state["proposed_action_id"], actor=WORKFLOW)
        return {
            "approval_id": proposal.approval_id,
            "approval_decision": None,
            "approval_rounds": rounds + 1,
        }

    def execute_refund_node(state: GraphState) -> dict[str, Any]:
        # Rechecks the approval, eligibility and balance; a rerun returns the same operation.
        result = gateway(WORKFLOW, {"execute_approved_refund"}).execute(
            "execute_approved_refund",
            {"proposal_id": state["proposed_action_id"]},
            call_id="execute_refund",
        )
        if not result["success"]:
            return {"route": "escalate", "validation_problems": [result["error_code"]]}
        operation = result["data"]
        return {
            "refund_operation_id": operation["operation_id"],
            "execution_status": operation["status"],
            "external_result_reference": operation["provider_reference"],
        }

    def verify_refund_node(state: GraphState) -> dict[str, Any]:
        """W3.11: the run reports nothing until the provider confirms the refund's outcome.

        Asks the provider by idempotency key a few times with a short backoff. A final status
        ends the check; anything else leaves the operation for the reconciliation sweep.
        """
        status = state.get("execution_status")
        reference = state.get("external_result_reference")
        for attempt in range(VERIFY_ATTEMPTS):
            if status in FINAL:
                break
            if attempt:
                rt.sleep(VERIFY_BACKOFF_SECONDS * attempt)
            try:
                with ctx.lock:
                    operation = confirm_refund(ctx.db, ctx.payments, state["proposed_action_id"])
            except ServiceFailure as failure:
                return {"route": "escalate", "validation_problems": [failure.error_code]}
            status, reference = operation.status, operation.provider_reference
        return {"execution_status": status, "external_result_reference": reference}

    def amount_text(state: GraphState) -> str:
        return f"₹{state['proposal']['amount_minor'] / 100:,.2f}"

    def finish_refunded_node(state: GraphState) -> dict[str, Any]:
        response = (
            f"Your refund of {amount_text(state)} for order {state['proposal']['order_id']} has "
            "been issued and the payment provider confirmed it (reference "
            f"{state.get('external_result_reference')})."
        )
        finish(
            state,
            run_status="completed",
            ticket_status="resolved",
            response=response,
            route="refunded",
        )
        return {"final_response": response}

    def finish_refund_failed_node(state: GraphState) -> dict[str, Any]:
        response = (
            f"Your refund of {amount_text(state)} for order {state['proposal']['order_id']} was "
            "approved, but the payment provider could not complete it, so no refund has been "
            "issued. A member of our support team will follow up on this ticket."
        )
        finish(
            state,
            run_status="escalated",
            ticket_status="escalated",
            response=response,
            route="refund_failed",
        )
        return {"final_response": response}

    def report_pending_node(state: GraphState) -> dict[str, Any]:
        # Claim no more than the provider confirmed; the run stays open until it does.
        response = (
            f"Your refund of {amount_text(state)} for order {state['proposal']['order_id']} was "
            "approved and sent to the payment provider, which has not confirmed it yet. We "
            "will update this ticket as soon as it does."
        )
        finish(
            state,
            run_status=None,
            ticket_status="open",
            response=response,
            route="refund_pending",
        )
        return {"final_response": response}

    def await_provider_node(state: GraphState) -> dict[str, Any]:
        # Nothing else happens here: on resume this node restarts from the top (W3.3).
        interrupt({"refund_operation_id": state.get("refund_operation_id")})
        return {}

    def finish_rejected_node(state: GraphState) -> dict[str, Any]:
        response = (
            "A support reviewer looked at your refund request for order "
            f"{state['proposal']['order_id']} and could not approve it. No refund has been "
            "issued. Reply on this ticket if you have more information for us."
        )
        finish(
            state,
            run_status="completed",
            ticket_status="resolved",
            response=response,
            route="rejected",
        )
        return {"final_response": response}

    def finish_answer_node(state: GraphState) -> dict[str, Any]:
        response = state["proposal"]["customer_message"]
        finish(
            state,
            run_status="completed",
            ticket_status="resolved",
            response=response,
            route="answer_only",
        )
        return {"final_response": response}

    def finish_approval_node(state: GraphState) -> dict[str, Any]:
        proposal = state["proposal"]
        rupees = proposal["amount_minor"] / 100
        response = (
            f"Thank you for your patience. Your refund request for order {proposal['order_id']} "
            f"(₹{rupees:,.2f}) has been checked against our refund policy and sent to a support "
            "reviewer for approval. No refund has been issued yet; we will update this ticket "
            "once the payment provider confirms it."
        )
        # The proposal service already moved the run to waiting_for_approval.
        finish(
            state,
            run_status=None,
            ticket_status="awaiting_approval",
            response=response,
            route="request_approval",
        )
        return {"final_response": response}

    def finish_escalate_node(state: GraphState) -> dict[str, Any]:
        response = (
            "Thank you for contacting us. Your request needs a member of our support team to "
            "review it, and they will follow up on this ticket."
        )
        finish(
            state,
            run_status="escalated",
            ticket_status="escalated",
            response=response,
            route="escalate",
        )
        return {"final_response": response, "route": "escalate"}

    def finish(
        state: GraphState, *, run_status: str | None, ticket_status: str, response: str, route: str
    ) -> None:
        """Record the outcome on the ticket and the run (code only, W2.12)."""
        category = state.get("complaint_category")
        gateway(WORKFLOW, {"update_ticket"}).execute(
            "update_ticket",
            {
                "ticket_id": ctx.ticket_id,
                "status": ticket_status,
                "note": state.get("summary"),
                "category": category,
            },
            call_id=f"finish_{route}",
        )
        with ctx.lock:
            if run_status is not None:
                set_run_status(ctx.db, ctx.run_id, run_status, actor=WORKFLOW)
            set_run_output(
                ctx.db,
                ctx.run_id,
                {
                    "final_response": response,
                    "route": route,
                    "category": category,
                    "order_id": state.get("order_id"),
                    "proposal_id": state.get("proposed_action_id"),
                    "approval_id": state.get("approval_id"),
                    "approval_decision": state.get("approval_decision"),
                    "refund_operation_id": state.get("refund_operation_id"),
                    "refund_status": state.get("execution_status"),
                    "policy_references": state.get("policy_references") or [],
                    "validation_problems": state.get("validation_problems") or [],
                    "stop_reason": state.get("stop_reason"),
                },
            )
            ctx.db.commit()
        # The trace's end of turn (W3.15, E§19): what the customer was told and why it stopped.
        halted_by = state.get("stop_reason")
        rt.recorder.record(
            WORKFLOW,
            "agent_finished",
            {
                "route": route,
                "stop_reason": AGENT_STOP.get(halted_by, "error") if halted_by else "model_final",
                "halted_by": halted_by,
                "run_status": run_status,
                "final_response": response,
            },
        )

    def clarification_question(state: GraphState) -> str:
        missing = state.get("missing_information") or []
        proposal = state.get("proposal") or {}
        if "order_id" not in missing and proposal.get("customer_message"):
            return proposal["customer_message"]
        if "order_id" in missing:
            with ctx.lock:
                orders = list_orders(ctx.db, ctx.customer, limit=5)
            listed = "; ".join(f"{o.id} ({', '.join(o.product_names)})" for o in orders)
            question = "Which order is this about? Please reply with the order ID (it starts "
            question += "with ord_)."
            return f"{question} Your recent orders: {listed}." if listed else question
        return "Could you share: " + ", ".join(missing) + "?"

    # --- Routing (code) ----------------------------------------------------------------

    def after_classify(state: GraphState) -> str | list[str]:
        if state.get("halted") or state.get("complaint_category") == "out_of_scope":
            return "finish_escalate"
        if rt.config.architecture == "single":
            return "single_agent"
        if state.get("complaint_category") == "refund_request":
            return "investigate_then_policy"
        return ["investigate", "policy"]

    def after_gather(state: GraphState) -> str:
        return "finish_escalate" if state.get("halted") else "check_missing"

    def after_check_missing(state: GraphState) -> str:
        if state.get("missing_information"):
            rounds = state.get("clarification_rounds") or 0
            return (
                "request_clarification" if rounds < MAX_CLARIFICATION_ROUNDS else "finish_escalate"
            )
        return "validate" if state.get("proposal") else "resolve"

    def after_resolve(state: GraphState) -> str:
        return "finish_escalate" if state.get("halted") else "validate"

    def after_validate(state: GraphState) -> str:
        route = state.get("route")
        if route == "ask_clarification":
            rounds = state.get("clarification_rounds") or 0
            return (
                "request_clarification" if rounds < MAX_CLARIFICATION_ROUNDS else "finish_escalate"
            )
        return {
            "request_approval": "persist_proposal",
            "answer_only": "finish_answer",
        }.get(route or "", "finish_escalate")

    def after_persist(state: GraphState) -> str:
        return "finish_escalate" if state.get("route") == "escalate" else "finish_approval"

    def after_decision(state: GraphState) -> str:
        if state.get("halted"):
            return "finish_escalate"
        decision = state.get("approval_decision")
        if decision is None:
            return "await_approval"  # woken without a decision: keep waiting
        return {
            "approved": "revalidate",
            "expired": "revalidate",
            "rejected": "finish_rejected",
        }.get(decision, "finish_escalate")  # invalidated elsewhere: a person looks at it

    def after_revalidate(state: GraphState) -> str:
        if state.get("halted") or state.get("route") == "escalate":
            return "finish_escalate"
        if state.get("revalidation") == "changed":
            return "finish_approval"
        approved = state.get("approval_decision") == "approved"
        return "execute_refund" if approved else "renew_approval"

    def after_renew(state: GraphState) -> str:
        failed = state.get("halted") or state.get("route") == "escalate"
        return "finish_escalate" if failed else "finish_approval"

    def after_execute(state: GraphState) -> str:
        failed = state.get("halted") or state.get("route") == "escalate"
        return "finish_escalate" if failed else "verify_refund"

    def after_verify(state: GraphState) -> str:
        if state.get("halted") or state.get("route") == "escalate":
            return "finish_escalate"
        return {"succeeded": "finish_refunded", "failed": "finish_refund_failed"}.get(
            state.get("execution_status") or "", "report_pending"
        )

    graph = StateGraph(GraphState)
    for name, node in [
        ("classify", classify_node),
        ("single_agent", single_agent_node),
        ("investigate", investigate_node),
        ("policy", policy_node),
        ("investigate_then_policy", investigate_then_policy_node),
        ("check_missing", check_missing_node),
        ("resolve", resolve_node),
        ("validate", validate_node),
        ("persist_proposal", persist_proposal_node),
        ("request_clarification", request_clarification_node),
        ("finish_answer", finish_answer_node),
        ("finish_approval", finish_approval_node),
        ("finish_escalate", finish_escalate_node),
        ("check_decision", check_decision_node),
        ("revalidate", revalidate_node),
        ("renew_approval", renew_approval_node),
        ("execute_refund", execute_refund_node),
        ("verify_refund", verify_refund_node),
        ("finish_refunded", finish_refunded_node),
        ("finish_refund_failed", finish_refund_failed_node),
        ("report_pending", report_pending_node),
        ("finish_rejected", finish_rejected_node),
    ]:
        graph.add_node(name, guarded(node))
    # Not guarded: interrupt() works by raising, and it must propagate to LangGraph.
    graph.add_node("await_customer", await_customer_node)
    graph.add_node("await_approval", await_approval_node)
    graph.add_node("await_provider", await_provider_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        after_classify,
        ["finish_escalate", "single_agent", "investigate_then_policy", "investigate", "policy"],
    )
    graph.add_conditional_edges("single_agent", after_gather, ["check_missing", "finish_escalate"])
    graph.add_conditional_edges(
        "investigate_then_policy", after_gather, ["check_missing", "finish_escalate"]
    )
    graph.add_node("join", lambda state: {})
    graph.add_edge(["investigate", "policy"], "join")
    graph.add_conditional_edges("join", after_gather, ["check_missing", "finish_escalate"])
    graph.add_conditional_edges(
        "check_missing",
        after_check_missing,
        ["request_clarification", "finish_escalate", "validate", "resolve"],
    )
    graph.add_conditional_edges("resolve", after_resolve, ["validate", "finish_escalate"])
    graph.add_conditional_edges(
        "validate",
        after_validate,
        ["persist_proposal", "finish_answer", "finish_escalate", "request_clarification"],
    )
    graph.add_conditional_edges(
        "persist_proposal", after_persist, ["finish_approval", "finish_escalate"]
    )
    graph.add_edge("request_clarification", "await_customer")
    graph.add_edge("await_customer", "classify")
    graph.add_edge("finish_approval", "await_approval")
    graph.add_edge("await_approval", "check_decision")
    graph.add_conditional_edges(
        "check_decision",
        after_decision,
        ["revalidate", "finish_rejected", "await_approval", "finish_escalate"],
    )
    graph.add_conditional_edges(
        "revalidate",
        after_revalidate,
        ["execute_refund", "renew_approval", "finish_approval", "finish_escalate"],
    )
    graph.add_conditional_edges(
        "renew_approval", after_renew, ["finish_approval", "finish_escalate"]
    )
    graph.add_conditional_edges(
        "execute_refund", after_execute, ["verify_refund", "finish_escalate"]
    )
    graph.add_conditional_edges(
        "verify_refund",
        after_verify,
        ["finish_refunded", "finish_refund_failed", "report_pending", "finish_escalate"],
    )
    graph.add_edge("report_pending", "await_provider")
    graph.add_edge("await_provider", "verify_refund")
    for name in (
        "finish_answer",
        "finish_escalate",
        "finish_refunded",
        "finish_refund_failed",
        "finish_rejected",
    ):
        graph.add_edge(name, END)
    return graph
