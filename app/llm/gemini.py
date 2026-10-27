"""Gemini through the official google-genai SDK (Q4, decided 2026-10-27: no LangChain).

Automatic function calling is off: the model only *requests* tool calls, and our tool gateway
decides whether to run them (W2.5).
"""

import uuid
from typing import Any

from google import genai
from google.genai import errors, types

from app.llm.types import Message, ModelError, ModelResponse, ToolCall, ToolSchema, Usage


class GeminiProvider:
    name = "gemini"

    def __init__(self, model_id: str, api_key: str, *, timeout_seconds: float = 60.0) -> None:
        if not model_id:
            raise ValueError("LLM_MODEL is not set; pin a Gemini model ID in .env (Q4)")
        if not api_key:
            raise ValueError("LLM_API_KEY is not set; add your Gemini key to .env")
        self.model_id = model_id
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

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
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if tools:
            config.tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.parameters,
                        )
                        for tool in tools
                    ]
                )
            ]
        if response_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = response_schema
        try:
            response = self.client.models.generate_content(
                model=self.model_id, contents=to_contents(messages), config=config
            )
        except errors.APIError as exc:
            code = int(exc.code or 0)
            retry_after, quota = rate_limit_details(exc.details)
            raise ModelError(
                f"HTTP_{code}",
                str(exc.message or exc),
                retryable=code == 429 or code >= 500,
                retry_after=retry_after,
                detail=quota,
            ) from None
        except (TimeoutError, OSError) as exc:
            raise ModelError("MODEL_TIMEOUT", str(exc), retryable=True) from None
        return from_response(response)


def rate_limit_details(details: Any) -> tuple[float | None, str | None]:
    """The retry delay and the exhausted quota from a 429 body (google.rpc RetryInfo and
    QuotaFailure), e.g. (37.0, "GenerateRequestsPerMinutePerProjectPerModel-FreeTier")."""
    error = details.get("error", details) if isinstance(details, dict) else {}
    retry_after: float | None = None
    quota: str | None = None
    for item in error.get("details", []) if isinstance(error, dict) else []:
        kind = str(item.get("@type", ""))
        if kind.endswith("RetryInfo"):
            delay = str(item.get("retryDelay", "")).removesuffix("s")
            try:
                retry_after = float(delay)
            except ValueError:
                pass
        elif kind.endswith("QuotaFailure"):
            violations = item.get("violations") or [{}]
            quota = violations[0].get("quotaId") or violations[0].get("quotaMetric")
    return retry_after, quota


def to_contents(messages: list[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        if message.role == "assistant" and message.raw is not None:
            contents.append(message.raw)
        elif message.role == "assistant":
            parts = [types.Part(text=message.text)] if message.text else []
            parts += [
                types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.arguments))
                for c in message.tool_calls
            ]
            contents.append(types.Content(role="model", parts=parts))
        elif message.role == "tool":
            part = types.Part(
                function_response=types.FunctionResponse(
                    id=message.tool_call_id, name=message.tool_name, response=message.tool_result
                )
            )
            # Consecutive tool results go back together, as one user turn.
            if contents and contents[-1].role == "user" and is_tool_turn(contents[-1]):
                contents[-1].parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
        else:
            contents.append(types.Content(role="user", parts=[types.Part(text=message.text)]))
    return contents


def is_tool_turn(content: types.Content) -> bool:
    return bool(content.parts) and all(p.function_response for p in content.parts)


def from_response(response: types.GenerateContentResponse) -> ModelResponse:
    candidate = response.candidates[0] if response.candidates else None
    parts = (
        candidate.content.parts
        if candidate and candidate.content and candidate.content.parts
        else []
    )
    text = "".join(p.text for p in parts if p.text and not p.thought)
    calls = [
        ToolCall(
            id=p.function_call.id or f"call_{uuid.uuid4().hex[:12]}",
            name=p.function_call.name or "",
            arguments=dict(p.function_call.args or {}),
        )
        for p in parts
        if p.function_call
    ]
    meta = response.usage_metadata
    usage = Usage(
        input_tokens=meta.prompt_token_count if meta else None,
        output_tokens=meta.candidates_token_count if meta else None,
        cached_input_tokens=meta.cached_content_token_count if meta else None,
        reasoning_tokens=meta.thoughts_token_count if meta else None,
    )
    return ModelResponse(
        text=text,
        tool_calls=calls,
        usage=usage,
        resolved_model=response.model_version,
        finish_reason=str(candidate.finish_reason)
        if candidate and candidate.finish_reason
        else None,
        raw=candidate.content if candidate else None,
    )
