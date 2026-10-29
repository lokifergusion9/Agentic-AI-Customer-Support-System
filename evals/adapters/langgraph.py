"""The LangGraph adapter: the real resolution agent under test (W4.7, E§10).

Wraps the W2.9 single agent and the W2.10 specialist design. The model, prompt version,
architecture and limits come from the `agent_configs` row in the context; the task's
`max_model_turns` caps the workflow's step limit (Q4). The ticket is opened from the agent
view as the customer, the run is started, every approval wait is handed to the reviewer
simulator and the run resumed on its thread, until it stops waiting.

Each attempt gets its own in-memory checkpointer: an interrupted attempt is never resumed,
it is replaced by a new attempt in a fresh environment (W4.8).
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from langgraph.checkpoint.memory import InMemorySaver

from app.agents.config import AgentConfig, build_provider
from app.llm.types import ModelProvider
from app.workflows.graph import AGENT_STOP
from app.workflows.runner import (
    WorkflowServices,
    resume_after_decision,
    resume_after_provider,
    start_resolution,
)
from evals.adapters.base import (
    AgentResult,
    AttemptCancelled,
    AttemptContext,
    open_ticket,
    resolve_customer,
)
from evals.configs import agent_config
from evals.task_format import AgentView

# Approval rounds the adapter will wait through; the workflow itself stops at three (W3.6).
MAX_APPROVAL_WAITS = 5


@dataclass
class LangGraphAdapter:
    # Tests pass a scripted model; by default the provider comes from the config (Gemini).
    provider_factory: Callable[[AgentConfig], ModelProvider] = build_provider

    async def run(self, task_input: AgentView, context: AttemptContext) -> AgentResult:
        if context.cancel.is_set():
            raise AttemptCancelled
        db = context.db
        config = agent_config(context.agent_config)
        config = replace(config, max_steps=min(config.max_steps, context.limits.max_model_turns))
        services = WorkflowServices(
            provider=self.provider_factory(config),
            config=config,
            payments=context.payments,
            shipping=context.shipping,
            ticketing=context.ticketing,
            embedder=context.embedder,
            checkpointer=InMemorySaver(),
            today=context.environment.as_of,
            sleep=context.sleep,
        )

        customer = resolve_customer(db, task_input.agent_input.customer_subject)
        ticket_id, order_refused = open_ticket(db, customer, task_input.agent_input)
        run = start_resolution(db, customer, ticket_id, services)

        for _ in range(MAX_APPROVAL_WAITS):
            if run.status != "waiting_for_approval":
                break
            approval_id = context.reviewer.pending_approval(db, run.id)
            if approval_id is None or context.cancel.is_set():
                break
            context.reviewer.decide(db, approval_id)
            run = resume_after_decision(db, approval_id, services)
        if run.status == "running" and not context.cancel.is_set():
            # Approved and sent, but the provider had not confirmed: ask it once more (W3.11).
            run = resume_after_provider(db, run.id, services)

        output = run.output or {}
        # The run stores why it halted (None for a normal end), as agent_finished reports it.
        halted_by = output.get("stop_reason")
        return AgentResult(
            final_response=output.get("final_response", ""),
            stop_reason=AGENT_STOP.get(halted_by, halted_by) if halted_by else "model_final",
            metadata={
                "workflow_run_id": run.id,
                "ticket_id": ticket_id,
                "run_status": run.status,
                "route": output.get("route"),
                "config_id": context.agent_config["id"],
                "model_id": config.model_id,
                "max_steps": config.max_steps,
                "ticket_order_refused": order_refused,
                "simulated_approvals": [d.as_event() for d in context.reviewer.decisions],
            },
        )
