"""The tool-using agent loop and validated structured output (W2.9, W2.15, E§11).

`run_tool_loop` lets the model request tools until it stops; every request goes through the
tool gateway (allowlist, validation, audit). `structured_output` asks for JSON matching a
Pydantic schema; invalid output gets one repair attempt, then the run stops and escalates.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agents.config import AgentConfig, Prompt
from app.llm.calls import RunBudget, RunRecorder, call_model
from app.llm.types import Message, ModelProvider, ToolSchema
from app.tools.base import ToolContext
from app.tools.gateway import ToolGateway

# Model turns per agent; the run-wide step and token budget applies on top (W2.15).
MAX_AGENT_TURNS = 8
REPAIR_ATTEMPTS = 1


class InvalidModelOutput(Exception):
    def __init__(self, schema: str, problems: list[str]) -> None:
        super().__init__(f"The model did not return valid {schema} JSON: {'; '.join(problems)}")
        self.schema = schema
        self.problems = problems


@dataclass
class AgentRuntime:
    """Everything an agent needs for one run; built by the workflow, never by the model."""

    provider: ModelProvider
    config: AgentConfig
    recorder: RunRecorder
    budget: RunBudget
    ctx: ToolContext
    sleep: Callable[[float], None] = time.sleep

    def call(self, *, actor: str, prompt: Prompt, messages: list[Message], **kwargs: Any):
        return call_model(
            self.provider,
            recorder=self.recorder,
            budget=self.budget,
            actor=actor,
            prompt_label=prompt.label,
            system=prompt.text,
            messages=messages,
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            sleep=self.sleep,
            **kwargs,
        )


@dataclass
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class LoopResult:
    messages: list[Message]
    tool_records: list[ToolRecord] = field(default_factory=list)
    final_text: str = ""
    # model_final | max_turns
    stop_reason: str = "model_final"


def run_tool_loop(
    rt: AgentRuntime,
    *,
    actor: str,
    prompt: Prompt,
    messages: list[Message],
    allowed: frozenset[str],
    max_turns: int = MAX_AGENT_TURNS,
) -> LoopResult:
    gateway = ToolGateway(rt.ctx, allowed, actor=actor)
    tools = [ToolSchema(**schema) for schema in gateway.schemas()]
    result = LoopResult(messages=list(messages))
    for _ in range(max_turns):
        response = rt.call(actor=actor, prompt=prompt, messages=result.messages, tools=tools)
        result.messages.append(response.as_message())
        if not response.tool_calls:
            result.final_text = response.text
            return result
        for call in response.tool_calls:
            rt.budget.take_step()
            outcome = gateway.execute(call.name, call.arguments, call_id=call.id)
            result.tool_records.append(ToolRecord(call.name, dict(call.arguments), outcome))
            result.messages.append(
                Message("tool", tool_call_id=call.id, tool_name=call.name, tool_result=outcome)
            )
    result.stop_reason = "max_turns"
    return result


def structured_output[M: BaseModel](
    rt: AgentRuntime,
    *,
    actor: str,
    prompt: Prompt,
    messages: list[Message],
    schema: type[M],
    instruction: str,
) -> M:
    """Ask for JSON matching `schema`; repair once on invalid output, then give up (W2.15)."""
    conversation = [*messages, Message("user", instruction)]
    problems: list[str] = []
    for attempt in range(1 + REPAIR_ATTEMPTS):
        response = rt.call(
            actor=actor,
            prompt=prompt,
            messages=conversation,
            response_schema=schema.model_json_schema(),
        )
        try:
            return schema.model_validate_json(strip_code_fence(response.text))
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(p) for p in error['loc']) or 'json'}: {error['msg']}"
                for error in exc.errors()[:5]
            ]
        rt.recorder.record(
            actor,
            "model_output_invalid",
            {"schema": schema.__name__, "attempt": attempt + 1, "problems": problems},
        )
        conversation = [
            *conversation,
            response.as_message(),
            Message(
                "user",
                "That reply did not match the JSON schema: "
                + "; ".join(problems)
                + ". Reply again with only the corrected JSON.",
            ),
        ]
    raise InvalidModelOutput(schema.__name__, problems)


def strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.DOTALL)
    return match.group(1) if match else text


def as_data(value: Any) -> str:
    """Render facts for a prompt as compact JSON."""
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
