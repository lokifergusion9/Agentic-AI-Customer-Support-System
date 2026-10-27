"""Recorded, budgeted model calls (W2.15, W2.16, E§11, E§21).

Every request to the model, including each retry, writes a `model_call` audit event with the
provider, configured and resolved model, prompt version, token usage as the provider reported
it (None when it did not), latency on a monotonic clock, attempt number and outcome. No prompt
or response text is stored.

The run budget caps model calls plus tool calls (MAX_WORKFLOW_STEPS) and tokens
(MAX_RUN_TOKENS). A local check is not a billing cap: a request already sent can still use
tokens, so the check runs before each call and actual usage is added after it (E§11).
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.llm.types import Message, ModelError, ModelProvider, ModelResponse, ToolSchema
from app.services.audit import record_event


class BudgetExceeded(Exception):
    def __init__(self, limit: str, message: str) -> None:
        super().__init__(message)
        self.limit = limit  # "max_steps" | "token_limit"


@dataclass
class RunBudget:
    max_steps: int
    max_tokens: int
    steps_used: int = 0
    tokens_used: int = 0

    def take_step(self) -> None:
        if self.steps_used >= self.max_steps:
            raise BudgetExceeded("max_steps", f"The run reached its limit of {self.max_steps} steps.")
        if self.tokens_used >= self.max_tokens:
            raise BudgetExceeded(
                "token_limit", f"The run reached its limit of {self.max_tokens} tokens."
            )
        self.steps_used += 1

    def add_usage(self, response: ModelResponse, messages: list[Message]) -> None:
        total = response.usage.total
        if total is None:
            # Unknown usage still counts against the budget, by a rough estimate; the audit
            # event keeps it as unknown (E§21: missing usage is not free).
            chars = sum(len(m.text) + len(str(m.tool_result or "")) for m in messages)
            total = (chars + len(response.text)) // 4
        self.tokens_used += total


@dataclass
class RunRecorder:
    """Writes audit events for one run. The lock serializes database access when two agents
    run in parallel threads (W2.10), since one SQLAlchemy session is not thread-safe."""

    db: Session
    run_id: str
    ticket_id: str
    lock: threading.RLock = field(default_factory=threading.RLock)

    def record(self, actor: str, event_type: str, payload: dict[str, Any]) -> None:
        with self.lock:
            record_event(
                self.db,
                actor=actor,
                event_type=event_type,
                payload=payload,
                ticket_id=self.ticket_id,
                run_id=self.run_id,
            )
            self.db.commit()


def call_model(
    provider: ModelProvider,
    *,
    recorder: RunRecorder,
    budget: RunBudget,
    actor: str,
    prompt_label: str,
    system: str,
    messages: list[Message],
    tools: list[ToolSchema] | None = None,
    response_schema: dict[str, Any] | None = None,
    max_output_tokens: int = 2048,
    temperature: float = 0.0,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> ModelResponse:
    attempt = 0
    while True:
        budget.take_step()
        attempt += 1
        started = time.monotonic()
        base = {
            "provider": provider.name,
            "model_id": provider.model_id,
            "prompt": prompt_label,
            "attempt": attempt,
            "tools_offered": len(tools or []),
            "structured_output": response_schema is not None,
        }
        try:
            response = provider.generate(
                system=system,
                messages=messages,
                tools=tools,
                response_schema=response_schema,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except ModelError as error:
            recorder.record(
                actor,
                "model_call",
                {
                    **base,
                    "outcome": "error",
                    "error_code": error.code,
                    "retryable": error.retryable,
                    "latency_ms": elapsed_ms(started),
                    # A failed request may still be billed; the provider did not say.
                    "input_tokens": None,
                    "output_tokens": None,
                },
            )
            if error.retryable and attempt <= max_retries:
                sleep(min(2 ** (attempt - 1), 8))
                continue
            raise
        budget.add_usage(response, messages)
        recorder.record(
            actor,
            "model_call",
            {
                **base,
                "outcome": "ok",
                "resolved_model": response.resolved_model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cached_input_tokens": response.usage.cached_input_tokens,
                "reasoning_tokens": response.usage.reasoning_tokens,
                "latency_ms": elapsed_ms(started),
                "finish_reason": response.finish_reason,
                "tool_calls": [call.name for call in response.tool_calls],
            },
        )
        return response


def elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)
