"""Deterministic stand-ins for slow or paid dependencies, shared by unit and integration tests."""

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.llm.types import Message, ModelError, ModelResponse, ToolCall, ToolSchema, Usage

DEFAULT_USAGE = Usage(100, 20)


def reply(
    text: str = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    *,
    usage: Usage | None = DEFAULT_USAGE,
) -> ModelResponse:
    calls = [
        ToolCall(f"call_{i}_{name}", name, args) for i, (name, args) in enumerate(tool_calls or [])
    ]
    return ModelResponse(text, calls, usage or Usage(), "fake-model-001", "STOP")


def json_reply(data: dict[str, Any], **kwargs: Any) -> ModelResponse:
    return reply(json.dumps(data), **kwargs)


@dataclass
class FakeCall:
    system: str
    messages: list[Message]
    tools: list[ToolSchema] | None
    response_schema: dict[str, Any] | None


Script = Callable[[FakeCall], ModelResponse | ModelError]


@dataclass
class ScriptedModel:
    """A deterministic model. `script` sees each request and returns a response or an error,
    so tests can react to which agent is calling (its system prompt) and what came back."""

    script: Script
    name: str = "fake"
    model_id: str = "fake-model"
    calls: list[FakeCall] = field(default_factory=list)

    def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> ModelResponse:
        call = FakeCall(system, list(messages), tools, response_schema)
        self.calls.append(call)
        result = self.script(call)
        if isinstance(result, ModelError):
            raise result
        return result


def sequence(*responses: ModelResponse | ModelError) -> Script:
    """A script that returns the given responses in order."""
    remaining = list(responses)
    return lambda call: remaining.pop(0)


class ListRecorder:
    """Collects audit events in memory, for tests that need no database."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def record(self, actor: str, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((actor, event_type, payload))


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors, with no model.

    Good enough to test ranking and filtering logic; real semantic quality is tested only in
    the `embedding_model` marked tests.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
