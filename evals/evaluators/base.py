"""What every evaluator reads and returns (W4.9, E§13, E§14).

An evaluator reads the evaluator view, the final-state snapshot (W4.4) and the trace, and
returns Scores in the score contract. It never calls a model: business state decides, not the
final message.
"""

from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Literal

from evals.environment import past_refund_keys
from evals.task_format import AgentInput, EvaluatorView

Status = Literal["pass", "fail", "partial", "not_applicable", "evaluation_error"]


@dataclass(frozen=True)
class Score:
    evaluator: str
    version: str
    dimension: str
    applicable: bool
    status: Status
    score: float | None
    reason_code: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Scorer:
    """Builds Scores for one evaluator."""

    def __init__(self, evaluator: str, version: str) -> None:
        self.evaluator = evaluator
        self.version = version

    def passed(self, dimension: str, **evidence: Any) -> Score:
        return Score(self.evaluator, self.version, dimension, True, "pass", 1.0, "OK", evidence)

    def failed(self, dimension: str, reason_code: str, **evidence: Any) -> Score:
        return Score(
            self.evaluator, self.version, dimension, True, "fail", 0.0, reason_code, evidence
        )

    def partial(self, dimension: str, score: float, reason_code: str, **evidence: Any) -> Score:
        return Score(
            self.evaluator, self.version, dimension, True, "partial", score, reason_code, evidence
        )

    def not_applicable(self, dimension: str, reason_code: str) -> Score:
        return Score(
            self.evaluator, self.version, dimension, False, "not_applicable", None, reason_code
        )

    def error(self, dimension: str, error: Exception) -> Score:
        return Score(
            self.evaluator,
            self.version,
            dimension,
            True,
            "evaluation_error",
            None,
            "EVALUATOR_RAISED",
            {"error": f"{type(error).__name__}: {error}"},
        )


@dataclass(frozen=True)
class ToolCall:
    sequence_number: int
    actor: str
    tool: str
    call_id: str
    args: dict[str, Any]
    outcome: str  # ok | error | rejected | missing (no result recorded)
    error_code: str | None
    result: dict[str, Any]

    @property
    def order_id(self) -> str | None:
        return self.args.get("order_id")


@dataclass
class AttemptRecord:
    """One attempt as the evaluators see it."""

    task: EvaluatorView
    agent_input: AgentInput
    snapshot: dict[str, Any]
    trace: list[dict[str, Any]]  # {sequence_number, actor, event_type, payload}
    final_response: str
    stop_reason: str
    workflow_run_id: str | None

    # --- Business state ---------------------------------------------------------------------

    def rows(self, table: str) -> list[dict[str, Any]]:
        return self.snapshot["support"][table]

    @cached_property
    def run(self) -> dict[str, Any] | None:
        return next(
            (r for r in self.rows("workflow_runs") if r["id"] == self.workflow_run_id), None
        )

    @cached_property
    def ticket(self) -> dict[str, Any] | None:
        if self.run is None:
            return None
        return next((t for t in self.rows("tickets") if t["id"] == self.run["ticket_id"]), None)

    @cached_property
    def proposals(self) -> list[dict[str, Any]]:
        """Proposals created by this attempt's run."""
        return [p for p in self.rows("action_proposals") if p["run_id"] == self.workflow_run_id]

    @cached_property
    def new_refunds(self) -> list[dict[str, Any]]:
        """Refunds the provider holds that were not in the fixture: this attempt's effects."""
        past = past_refund_keys(self.task.environment_fixture)
        return [
            r for r in self.snapshot["providers"]["refunds"] if r["idempotency_key"] not in past
        ]

    @cached_property
    def confirmed_refunds(self) -> list[dict[str, Any]]:
        return [r for r in self.new_refunds if r["status"] == "succeeded"]

    @cached_property
    def owner_of(self) -> dict[str, str]:
        return {o.id: o.customer_id for o in self.task.environment_fixture.orders}

    def owns(self, order_id: str) -> bool:
        return self.owner_of.get(order_id) == self.customer_id

    @cached_property
    def customer_id(self) -> str:
        return self.task.evaluation.expected_customer_id

    @cached_property
    def named_orders(self) -> set[str]:
        """Order IDs the customer gave: checking those is expected, even someone else's."""
        from evals.adapters.base import ORDER_ID_IN_TEXT

        named = set(ORDER_ID_IN_TEXT.findall(self.agent_input.user_message))
        if self.agent_input.order_id:
            named.add(self.agent_input.order_id)
        return named

    # --- Trace ------------------------------------------------------------------------------

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.trace if e["event_type"] == event_type]

    @cached_property
    def tool_calls(self) -> list[ToolCall]:
        return pair_tool_calls(self.trace)

    @cached_property
    def approvals_granted(self) -> list[dict[str, Any]]:
        return [
            e for e in self.events("approval_decided") if e["payload"]["decision"] == "approved"
        ]

    def steps(self) -> list[tuple[int, str]]:
        """(sequence number, step) for every executed tool call and approval milestone."""
        steps = [(c.sequence_number, c.tool) for c in self.tool_calls if c.outcome == "ok"]
        steps += [
            (e["sequence_number"], "approval_requested") for e in self.events("proposal_created")
        ]
        for event in self.events("approval_decided"):
            decision = event["payload"]["decision"]
            name = {"approved": "approval_granted", "rejected": "approval_rejected"}.get(
                decision, "approval_expired"
            )
            steps.append((event["sequence_number"], name))
        return sorted(steps)


def pair_tool_calls(trace: list[dict[str, Any]]) -> list[ToolCall]:
    """Each tool_requested paired with its tool_finished or tool_rejected, in order."""
    open_calls: dict[tuple[str, str], list[dict[str, Any]]] = {}
    results: dict[int, dict[str, Any]] = {}
    for event in trace:
        payload = event["payload"]
        if "tool" not in payload:
            continue
        key = (payload["tool"], payload.get("call_id", ""))
        if event["event_type"] == "tool_requested":
            open_calls.setdefault(key, []).append(event)
        elif event["event_type"] in ("tool_finished", "tool_rejected") and open_calls.get(key):
            results[open_calls[key].pop(0)["sequence_number"]] = event
    calls = []
    for event in (e for e in trace if e["event_type"] == "tool_requested"):
        payload = event["payload"]
        result = results.get(event["sequence_number"])
        if result is None:
            outcome, payload_result = "missing", {}
        elif result["event_type"] == "tool_rejected":
            outcome, payload_result = "rejected", result["payload"]
        else:
            payload_result = result["payload"]
            outcome = "ok" if payload_result.get("status") == "ok" else "error"
        args = payload.get("args")
        calls.append(
            ToolCall(
                sequence_number=event["sequence_number"],
                actor=event["actor"],
                tool=payload["tool"],
                call_id=payload.get("call_id", ""),
                args=args if isinstance(args, dict) else {},
                outcome=outcome,
                error_code=payload_result.get("error_code"),
                result=payload_result,
            )
        )
    return calls
