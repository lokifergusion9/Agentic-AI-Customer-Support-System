"""The agents: classifier, investigation, policy, resolution, and the single-agent baseline
(W2.8, W2.9, W2.10). Each gets only the tools it needs.

| Agent         | Tools                         | Output             |
|---------------|-------------------------------|--------------------|
| classifier    | none                          | Classification     |
| investigation | read tools, no search_policy  | EvidenceSummary    |
| policy        | search_policy only            | PolicyFinding      |
| resolution    | none (proposal is persisted   | ResolutionProposal |
|               | by code after validation)     |                    |
| single agent  | read tools + search_policy    | ResolutionProposal |
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

from app.agents.loop import AgentRuntime, ToolRecord, as_data, run_tool_loop, structured_output
from app.llm.types import Message
from app.schemas.agents import Classification, EvidenceSummary, PolicyFinding, ResolutionProposal
from app.tools.registry import READ_TOOLS

INVESTIGATION_TOOLS = READ_TOOLS - {"search_policy"}
POLICY_TOOLS = frozenset({"search_policy"})
SINGLE_AGENT_TOOLS = READ_TOOLS

SUMMARIZE = "Stop calling tools now and reply with the JSON object described in your instructions."


@dataclass
class AgentOutput[T]:
    output: T
    tool_records: list[ToolRecord]


def customer_block(messages: list[str]) -> str:
    """Customer text, fenced as data (it is untrusted input)."""
    body = "\n---\n".join(messages)
    return f"<customer_messages>\n{body}\n</customer_messages>"


def case_brief(state: dict[str, Any], today: date | None) -> str:
    lines = [customer_block(state.get("customer_messages") or [state["customer_message"]])]
    if state.get("complaint_category"):
        lines.append(f"Complaint category: {state['complaint_category']}")
    if state.get("order_id"):
        lines.append(f"Order ID on the ticket: {state['order_id']}")
    lines.append(f"Ticket ID: {state['ticket_id']}")
    lines.append(f"Today's date (store time): {(today or date.today()).isoformat()}")
    return "\n".join(lines)


def classify(rt: AgentRuntime, state: dict[str, Any]) -> Classification:
    prompt = rt.config.prompt("classifier")
    return structured_output(
        rt,
        actor="agent:classifier",
        prompt=prompt,
        messages=[],
        schema=Classification,
        instruction=customer_block(state.get("customer_messages") or [state["customer_message"]]),
    )


def investigate(rt: AgentRuntime, state: dict[str, Any]) -> AgentOutput[EvidenceSummary]:
    prompt = rt.config.prompt("investigation")
    loop = run_tool_loop(
        rt,
        actor="agent:investigation",
        prompt=prompt,
        messages=[Message("user", case_brief(state, rt.ctx.today))],
        allowed=INVESTIGATION_TOOLS,
    )
    summary = structured_output(
        rt,
        actor="agent:investigation",
        prompt=prompt,
        messages=loop.messages,
        schema=EvidenceSummary,
        instruction=SUMMARIZE,
    )
    return AgentOutput(summary, loop.tool_records)


def find_policy(
    rt: AgentRuntime, state: dict[str, Any], *, facts: dict[str, Any] | None = None
) -> AgentOutput[PolicyFinding]:
    prompt = rt.config.prompt("policy")
    brief = case_brief(state, rt.ctx.today)
    if facts:
        # Facts gathered first when the policy version depends on them (W2.10).
        brief += f"\nVerified facts: {as_data(facts)}"
    loop = run_tool_loop(
        rt,
        actor="agent:policy",
        prompt=prompt,
        messages=[Message("user", brief)],
        allowed=POLICY_TOOLS,
        max_turns=4,
    )
    finding = structured_output(
        rt,
        actor="agent:policy",
        prompt=prompt,
        messages=loop.messages,
        schema=PolicyFinding,
        instruction=SUMMARIZE,
    )
    return AgentOutput(finding, loop.tool_records)


def resolve(rt: AgentRuntime, state: dict[str, Any]) -> ResolutionProposal:
    prompt = rt.config.prompt("resolution")
    payload = "\n".join(
        [
            case_brief(state, rt.ctx.today),
            f"Verified facts: {as_data(state.get('verified_facts') or {})}",
            f"Investigation summary: {as_data(state.get('evidence') or {})}",
            f"Policy findings: {as_data(state.get('policy_finding') or {})}",
            f"Policy citations retrieved: {as_data(state.get('policy_references') or [])}",
        ]
    )
    return structured_output(
        rt,
        actor="agent:resolution",
        prompt=prompt,
        messages=[],
        schema=ResolutionProposal,
        instruction=payload,
    )


def single_agent(rt: AgentRuntime, state: dict[str, Any]) -> AgentOutput[ResolutionProposal]:
    """The W2.9 baseline: one agent gathers facts and policy, then proposes."""
    prompt = rt.config.prompt("single_agent")
    loop = run_tool_loop(
        rt,
        actor="agent:single",
        prompt=prompt,
        messages=[Message("user", case_brief(state, rt.ctx.today))],
        allowed=SINGLE_AGENT_TOOLS,
        max_turns=10,
    )
    proposal = structured_output(
        rt,
        actor="agent:single",
        prompt=prompt,
        messages=loop.messages,
        schema=ResolutionProposal,
        instruction="Stop calling tools now and reply with your proposal as JSON matching "
        "the schema.",
    )
    return AgentOutput(proposal, loop.tool_records)
