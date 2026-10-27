"""Provider-neutral messages and responses, so agents never depend on one SDK (E§6, E§10)."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For role "tool": which call this answers, and the structured result.
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_result: dict[str, Any] | None = None
    # The provider's own form of an assistant turn, replayed verbatim on the next request
    # (Gemini needs its thought signatures back). Never inspected by agent code.
    raw: Any = None


@dataclass(frozen=True)
class Usage:
    """Token counts as the provider reports them. None means unknown, never zero (E§21)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def total(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens + (self.reasoning_tokens or 0)


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    # The model version the provider says it used, when it says so.
    resolved_model: str | None
    finish_reason: str | None
    raw: Any = None

    def as_message(self) -> Message:
        return Message("assistant", self.text, list(self.tool_calls), raw=self.raw)


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: dict[str, Any]


class ModelError(Exception):
    """A failed model request. Retryable covers rate limits (429), 5xx and timeouts.

    `retry_after` is the wait the provider asked for, in seconds, when it said; `detail` names
    what failed, e.g. the exhausted quota (per minute or per day).
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        retry_after: float | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        self.detail = detail


class ModelProvider(Protocol):
    name: str
    model_id: str

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelResponse: ...
