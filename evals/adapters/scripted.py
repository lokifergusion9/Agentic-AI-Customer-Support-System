"""The scripted adapter: a deterministic fake agent for testing the platform (W4.6, E§10).

It calls the same typed tools as the real agent, through the same ToolGateway, so ownership,
eligibility and approval checks apply to it exactly as they do to a model. It costs nothing
and needs no API key; tests and `make smoke` use it.

`correct` is a small rule-based agent that works only from what the tools return (it never
sees expected outcomes). The other scripts are the same agent with one deliberate fault each,
so the evaluators can be shown to catch it (W4.9 gate):

- refund_without_approval: proposes, then executes before any reviewer decision
- other_customer_order:    looks up another customer's order instead of the one given
- false_success_claim:     tells the customer the refund was issued, having done nothing
- duplicate_refund:        after a confirmed refund, tries to refund the same order again
- ignore_tool_failure:     carries on after a failed tool call as if it had succeeded
"""

import re
from dataclasses import dataclass
from typing import Any

from app.auth.customer import AuthenticatedCustomer
from app.services.audit import record_event
from app.services.errors import ServiceFailure
from app.services.refunds import FINAL, confirm_refund
from app.services.runs import set_run_output, set_run_status, start_run
from app.tools.base import ToolContext
from app.tools.gateway import ToolGateway
from app.tools.registry import TOOLS
from evals.adapters.base import (
    AgentResult,
    AttemptCancelled,
    AttemptContext,
    open_ticket,
    order_in_message,
    resolve_customer,
)
from evals.task_format import AgentView

SCRIPTS = (
    "correct",
    "refund_without_approval",
    "other_customer_order",
    "false_success_claim",
    "duplicate_refund",
    "ignore_tool_failure",
)
ACTOR = "agent:scripted"
REFUND_WORDS = re.compile(r"refund|money back|reimburs", re.IGNORECASE)
ESCALATED = (
    "Thank you for contacting us. Your request needs a member of our support team to review "
    "it, and they will follow up on this ticket."
)
# Final states: (route, run status, ticket status).
ENDINGS = {
    "resolve": ("completed", "resolved"),
    "refuse": ("completed", "resolved"),
    "escalate": ("escalated", "escalated"),
    "clarify": ("waiting_for_customer", "awaiting_customer"),
}


class StopAttempt(Exception):
    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason


@dataclass
class ScriptedAdapter:
    script: str = "correct"
    # The order other_customer_order looks up instead (Bob's ord_1003 in the seed data).
    other_order_id: str = "ord_1003"

    def __post_init__(self) -> None:
        if self.script not in SCRIPTS:
            raise ValueError(f"unknown script {self.script!r}; scripts: {', '.join(SCRIPTS)}")

    async def run(self, task_input: AgentView, context: AttemptContext) -> AgentResult:
        db = context.db
        customer = resolve_customer(db, task_input.agent_input.customer_subject)
        ticket_id, order_refused = open_ticket(db, customer, task_input.agent_input)
        run = start_run(db, customer, ticket_id)
        db.commit()
        agent = ScriptedAgent(self, task_input, context, customer, run.id, ticket_id)
        try:
            route, response = agent.resolve()
            stop_reason = "model_final"
        except StopAttempt as stop:
            route, response, stop_reason = "escalate", ESCALATED, stop.stop_reason
        except AttemptCancelled:
            db.rollback()
            set_run_status(db, run.id, "failed", actor=ACTOR)
            agent.finish_run("failed", "", "cancelled")
            return agent.result("", "cancelled", order_refused)
        agent.finish(route, response, stop_reason)
        return agent.result(response, stop_reason, order_refused)


class ScriptedAgent:
    def __init__(
        self,
        adapter: ScriptedAdapter,
        task_input: AgentView,
        context: AttemptContext,
        customer: AuthenticatedCustomer,
        run_id: str,
        ticket_id: str,
    ) -> None:
        self.script = adapter.script
        self.other_order_id = adapter.other_order_id
        self.input = task_input.agent_input
        self.context = context
        self.run_id = run_id
        self.ticket_id = ticket_id
        tool_context = ToolContext(
            db=context.db,
            customer=customer,
            run_id=run_id,
            ticket_id=ticket_id,
            payments=context.payments,
            shipping=context.shipping,
            ticketing=context.ticketing,
            embedder=context.embedder,
            today=context.environment.as_of,
            sleep=context.sleep,
        )
        self.gateway = ToolGateway(tool_context, frozenset(TOOLS), actor=ACTOR)

    # --- Tools ------------------------------------------------------------------------------

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        if self.context.cancel.is_set():
            raise AttemptCancelled
        if self.gateway.calls >= self.context.limits.max_tool_calls:
            raise StopAttempt("max_tool_calls")
        return self.gateway.execute(name, arguments, call_id=f"s{self.gateway.calls + 1}")

    def failed(self, result: dict[str, Any]) -> bool:
        """A failed call; ignore_tool_failure pretends it went through."""
        return not result["success"] and self.script != "ignore_tool_failure"

    # --- The agent --------------------------------------------------------------------------

    def resolve(self) -> tuple[str, str]:
        order_id = order_in_message(self.input)
        if self.script == "other_customer_order":
            order_id = self.other_order_id
        if order_id is None:
            return self.ask_for_order()
        order = self.call("get_order", order_id=order_id)
        if not order["success"]:
            if order["error_code"] == "ORDER_NOT_FOUND":
                return self.ask_for_order()
            if self.failed(order):
                return "escalate", ESCALATED
        if REFUND_WORDS.search(self.input.user_message):
            return self.refund(order_id, order.get("data") or {})
        return self.order_status(order_id)

    def ask_for_order(self) -> tuple[str, str]:
        listed = self.call("list_orders")
        orders = listed["data"] if listed["success"] else []
        question = (
            "Which order is this about? Please reply with the order ID (it starts with ord_)."
        )
        if orders:
            question += " Your recent orders: " + "; ".join(
                f"{o['id']} ({', '.join(o['product_names'])})" for o in orders
            )
        return "clarify", question

    def order_status(self, order_id: str) -> tuple[str, str]:
        shipment = self.call("get_shipment_status", order_id=order_id)
        if not shipment["success"]:
            if self.failed(shipment):
                return "escalate", ESCALATED
            return "resolve", f"Your order {order_id} is on its way and will arrive soon."
        data = shipment["data"]
        status = data["status"].replace("_", " ")
        updated = data["last_updated_at"][:10]
        return "resolve", (
            f"Your order {order_id} is {status}, according to the courier's last update on "
            f"{updated}."
        )

    def refund(self, order_id: str, order: dict[str, Any]) -> tuple[str, str]:
        ret = self.call("get_return_status", order_id=order_id)
        payment = self.call("get_payment_status", order_id=order_id)
        if self.failed(ret) or self.failed(payment):
            return "escalate", ESCALATED
        if self.script == "false_success_claim":
            return "resolve", f"Good news: your refund for order {order_id} has been issued."

        ret_data, pay_data = ret.get("data") or {}, payment.get("data") or {}
        refunds = [r for r in pay_data.get("refunds", []) if r["status"] != "failed"]
        if any(r["status"] == "pending" for r in refunds):
            return "refuse", (
                f"A refund for order {order_id} is already being processed by the payment "
                "provider, so no new refund was created."
            )
        if refunds:
            return "refuse", f"Order {order_id} has already been refunded."
        received = ret_data.get("status") in ("received", "closed")
        if self.script != "ignore_tool_failure" and not received:
            return "refuse", (
                f"We can refund order {order_id} once our warehouse has received your return. "
                "It has not arrived yet."
            )
        if self.script != "ignore_tool_failure" and ret_data.get("inspection_status") != "passed":
            return "refuse", (
                f"The returned item from order {order_id} has not passed inspection, so a "
                "refund cannot be issued yet."
            )

        amount = sum(item["paid_amount_minor"] for item in order.get("items", []))
        evidence = [ref for ref in (ret_data.get("return_id"), pay_data.get("payment_id")) if ref]
        proposed = self.call(
            "create_refund_proposal",
            order_id=order_id,
            amount_minor=amount,
            reason="Return received and inspection passed",
            evidence_refs=evidence,
        )
        if not proposed["success"]:
            if self.failed(proposed):
                return "escalate", ESCALATED
            return "resolve", f"Your refund for order {order_id} has been issued."
        proposal = proposed["data"]

        if self.script == "refund_without_approval":
            self.call("execute_approved_refund", proposal_id=proposal["proposal_id"])
            return "resolve", f"Your refund for order {order_id} has been issued."

        decision = self.context.reviewer.decide(self.context.db, proposal["approval_id"])
        if decision.outcome == "rejected":
            return "refuse", (
                f"A support reviewer looked at your refund request for order {order_id} and "
                "could not approve it. No refund has been issued."
            )
        if decision.outcome != "approved":
            return "escalate", ESCALATED

        executed = self.call("execute_approved_refund", proposal_id=proposal["proposal_id"])
        status = executed["data"]["status"] if executed["success"] else None
        if status not in ("succeeded", "failed") and executed["success"]:
            checked = self.call("get_refund_status", operation_id=executed["data"]["operation_id"])
            status = checked["data"]["local_status"] if checked["success"] else None
        if status not in FINAL and executed["success"]:
            # The workflow's verify step is code, not the agent: it asks the provider by
            # idempotency key before anything is claimed (graph.py verify_refund_node).
            try:
                status = confirm_refund(
                    self.context.db, self.context.payments, proposal["proposal_id"]
                ).status
            except ServiceFailure:
                status = None
        if status != "succeeded":
            return "escalate", ESCALATED

        if self.script == "duplicate_refund":
            self.call(
                "create_refund_proposal",
                order_id=order_id,
                amount_minor=amount,
                reason="Refund again",
                evidence_refs=evidence,
            )
        rupees = f"₹{amount / 100:,.2f}"
        return "resolve", (
            f"Your refund of {rupees} for order {order_id} has been confirmed by the payment "
            "provider. It usually reaches your account in 5 to 7 business days."
        )

    # --- Ending -----------------------------------------------------------------------------

    def finish(self, route: str, response: str, stop_reason: str) -> None:
        run_status, ticket_status = ENDINGS[route]
        self.call_unlimited("update_ticket", ticket_id=self.ticket_id, status=ticket_status)
        set_run_status(self.context.db, self.run_id, run_status, actor=ACTOR)
        self.finish_run(route, response, stop_reason)

    def call_unlimited(self, name: str, **arguments: Any) -> dict[str, Any]:
        # Closing the ticket is the workflow's own step, not the agent's: no limit applies.
        return self.gateway.execute(name, arguments, call_id=f"s{self.gateway.calls + 1}")

    def finish_run(self, route: str, response: str, stop_reason: str) -> None:
        output = {"route": route, "stop_reason": stop_reason, "final_response": response}
        set_run_output(self.context.db, self.run_id, output)
        record_event(
            self.context.db,
            actor=ACTOR,
            event_type="agent_finished",
            payload=output,
            ticket_id=self.ticket_id,
            run_id=self.run_id,
        )
        self.context.db.commit()

    def result(self, response: str, stop_reason: str, order_refused: bool) -> AgentResult:
        return AgentResult(
            final_response=response,
            stop_reason=stop_reason,
            metadata={
                "workflow_run_id": self.run_id,
                "ticket_id": self.ticket_id,
                "script": self.script,
                "tool_calls": self.gateway.calls,
                "ticket_order_refused": order_refused,
                "simulated_approvals": [d.as_event() for d in self.context.reviewer.decisions],
            },
        )
