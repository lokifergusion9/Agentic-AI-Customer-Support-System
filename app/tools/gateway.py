"""The tool gateway: where every tool call is checked and recorded (W2.4, W2.5, E§12).

For each call:

1. Record the attempt (`tool_requested`).
2. Look the tool up in the caller's allowlist (each agent gets only the tools it needs).
3. Parse and validate the arguments against the tool's Pydantic model.
4. Run the tool with server-side identity from the ToolContext.
5. Record the outcome: `tool_finished` when it ran, `tool_rejected` when it was refused
   (unknown tool, not allowed, invalid arguments, or a non-retryable service refusal such as
   ORDER_NOT_FOUND or NOT_ELIGIBLE). A refused attempt and an executed effect stay distinct.

The model always gets a structured result back, never an exception.
"""

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from app.services.audit import record_event
from app.services.errors import ServiceFailure
from app.tools.base import ToolContext, failure, ok
from app.tools.registry import TOOLS


def dump(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, list):
        return [dump(item) for item in result]
    return result


# Waits before retrying a read tool after a temporary provider failure (W3.16).
READ_RETRY_DELAYS = (0.5, 1.0)


class ToolGateway:
    def __init__(self, ctx: ToolContext, allowed: frozenset[str], *, actor: str) -> None:
        unknown = allowed - TOOLS.keys()
        if unknown:
            raise ValueError(f"unknown tools in allowlist: {sorted(unknown)}")
        self.ctx = ctx
        self.allowed = allowed
        # Who is calling, e.g. agent:investigation or system:workflow.
        self.actor = actor
        self.calls = 0

    def schemas(self) -> list[dict[str, Any]]:
        """Name, description and JSON schema of each allowed tool, for the model."""
        return [
            {
                "name": name,
                "description": TOOLS[name].description,
                "parameters": TOOLS[name].json_schema(),
            }
            for name in sorted(self.allowed)
        ]

    def execute(
        self, name: str, raw_arguments: dict[str, Any] | str | None, *, call_id: str
    ) -> dict[str, Any]:
        with self.ctx.lock:
            return self._execute(name, raw_arguments, call_id)

    def _execute(
        self, name: str, raw_arguments: dict[str, Any] | str | None, call_id: str
    ) -> dict[str, Any]:
        self.calls += 1
        started = time.monotonic()
        arguments = raw_arguments if raw_arguments is not None else {}
        spec = TOOLS.get(name)
        self.record("tool_requested", name, call_id, args=self.redact(name, arguments))

        if spec is None or name not in self.allowed:
            return self.reject(
                name,
                call_id,
                started,
                ServiceFailure("TOOL_NOT_ALLOWED", f"The tool {name!r} is not available here."),
            )
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            args = spec.args_model.model_validate(arguments)
        except (ValueError, ValidationError) as exc:
            return self.reject(
                name,
                call_id,
                started,
                ServiceFailure("INVALID_ARGUMENTS", validation_summary(exc)),
            )

        attempt = 0
        while True:
            try:
                result = ok(dump(spec.handler(self.ctx, args)))
                break
            except ServiceFailure as error:
                if not error.retryable:
                    return self.reject(name, call_id, started, error)
                # A provider timeout or outage: the tool ran but did not get an answer.
                self.ctx.db.rollback()
                if spec.access == "read" and attempt < len(READ_RETRY_DELAYS):
                    # Reads change nothing, so they are retried here, a bounded number of
                    # times with backoff (HLD 5b); writes have their own idempotent retries.
                    self.record(
                        "tool_retried",
                        name,
                        call_id,
                        attempt=attempt + 1,
                        error_code=error.error_code,
                    )
                    self.ctx.sleep(READ_RETRY_DELAYS[attempt])
                    attempt += 1
                    continue
                self.record(
                    "tool_finished",
                    name,
                    call_id,
                    status="error",
                    error_code=error.error_code,
                    retryable=True,
                    attempts=attempt + 1,
                    duration_ms=elapsed_ms(started),
                )
                return failure(error)

        self.record(
            "tool_finished",
            name,
            call_id,
            status="ok",
            duration_ms=elapsed_ms(started),
            **self.result_summary(name, result["data"]),
        )
        return result

    def reject(
        self, name: str, call_id: str, started: float, error: ServiceFailure
    ) -> dict[str, Any]:
        self.ctx.db.rollback()
        self.record(
            "tool_rejected",
            name,
            call_id,
            error_code=error.error_code,
            duration_ms=elapsed_ms(started),
        )
        return failure(error)

    def record(self, event_type: str, tool: str, call_id: str, **payload: Any) -> None:
        record_event(
            self.ctx.db,
            actor=self.actor,
            event_type=event_type,
            payload={"tool": tool, "call_id": call_id, **payload},
            ticket_id=self.ctx.ticket_id,
            run_id=self.ctx.run_id,
        )
        self.ctx.db.commit()

    def redact(self, name: str, arguments: Any) -> Any:
        """IDs, amounts and dates are kept; free text is reduced to its length."""
        spec = TOOLS.get(name)
        if not isinstance(arguments, dict):
            return {"unparsed_chars": len(str(arguments))}
        free_text = spec.free_text_args if spec else frozenset()
        return {
            key: {"chars": len(str(value))} if key in free_text else value
            for key, value in arguments.items()
        }

    @staticmethod
    def result_summary(name: str, data: Any) -> dict[str, Any]:
        if name == "search_policy":
            return {"search_status": data["status"], "citations": data["citations"]}
        if name == "create_refund_proposal":
            return {"proposal_id": data["proposal_id"], "approval_id": data["approval_id"]}
        if name == "execute_approved_refund":
            return {"operation_id": data["operation_id"], "refund_status": data["status"]}
        return {}


def elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def validation_summary(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        problems = [
            f"{'.'.join(str(p) for p in error['loc']) or 'arguments'}: {error['msg']}"
            for error in exc.errors()
        ]
        return "Invalid arguments: " + "; ".join(problems)
    return "Arguments are not valid JSON."
