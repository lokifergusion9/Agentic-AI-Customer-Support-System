"""Recorded, budgeted model calls and the Gemini message conversion (W2.15, W2.16)."""

import pytest
from google.genai import types

from app.llm.calls import BudgetExceeded, RunBudget, call_model
from app.llm.gemini import from_response, rate_limit_details, to_contents
from app.llm.types import Message, ModelError, ToolCall, Usage
from tests.fakes import ListRecorder, ScriptedModel, reply, sequence


def call(model: ScriptedModel, recorder: ListRecorder, budget: RunBudget, **kwargs):
    return call_model(
        model,
        recorder=recorder,  # type: ignore[arg-type]
        budget=budget,
        actor="agent:test",
        prompt_label="test_v1",
        system="You are a test.",
        messages=[Message("user", "hello")],
        sleep=lambda seconds: None,
        **kwargs,
    )


def test_every_call_records_usage_and_model_but_no_text() -> None:
    recorder, budget = ListRecorder(), RunBudget(max_steps=5, max_tokens=1000)
    model = ScriptedModel(sequence(reply("secret answer", usage=Usage(120, 30, 10, 5))))

    response = call(model, recorder, budget)

    assert response.text == "secret answer"
    [(actor, event, payload)] = recorder.events
    assert (actor, event) == ("agent:test", "model_call")
    assert payload["outcome"] == "ok" and payload["attempt"] == 1
    assert payload["model_id"] == "fake-model"
    assert payload["resolved_model"] == "fake-model-001"
    assert payload["prompt"] == "test_v1"
    assert (payload["input_tokens"], payload["output_tokens"]) == (120, 30)
    assert (payload["cached_input_tokens"], payload["reasoning_tokens"]) == (10, 5)
    assert payload["latency_ms"] >= 0
    assert "secret" not in str(payload) and "hello" not in str(payload)
    assert (budget.steps_used, budget.tokens_used) == (1, 155)


def test_rate_limit_is_retried_and_every_attempt_recorded() -> None:
    recorder, budget = ListRecorder(), RunBudget(max_steps=5, max_tokens=1000)
    model = ScriptedModel(sequence(ModelError("HTTP_429", "quota", retryable=True), reply("ok")))

    call(model, recorder, budget)

    outcomes = [(p["attempt"], p["outcome"], p.get("error_code")) for _, _, p in recorder.events]
    assert outcomes == [(1, "error", "HTTP_429"), (2, "ok", None)]
    assert recorder.events[0][2]["input_tokens"] is None
    assert budget.steps_used == 2


def test_provider_retry_delay_is_honored() -> None:
    recorder, budget = ListRecorder(), RunBudget(max_steps=5, max_tokens=1000)
    waits: list[float] = []
    model = ScriptedModel(
        sequence(
            ModelError(
                "HTTP_429",
                "quota",
                retryable=True,
                retry_after=37.0,
                detail="GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
            ),
            reply("ok"),
        )
    )

    call_model(
        model,
        recorder=recorder,
        budget=budget,
        actor="agent:test",  # type: ignore[arg-type]
        prompt_label="test_v1",
        system="s",
        messages=[Message("user", "hi")],
        sleep=waits.append,
    )

    assert waits == [37.0]
    failed = recorder.events[0][2]
    assert failed["retry_after_s"] == 37.0
    assert failed["error_detail"].startswith("GenerateRequestsPerMinute")


def test_a_wait_longer_than_a_minute_ends_the_call() -> None:
    model = ScriptedModel(
        sequence(ModelError("HTTP_429", "daily quota", retryable=True, retry_after=3600.0))
    )
    with pytest.raises(ModelError, match="daily quota"):
        call(model, ListRecorder(), RunBudget(max_steps=5, max_tokens=1000))
    assert len(model.calls) == 1


def test_rate_limit_details_are_read_from_the_error_body() -> None:
    body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"},
            ],
        }
    }
    assert rate_limit_details(body) == (12.0, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    assert rate_limit_details(None) == (None, None)


def test_retries_stop_at_the_limit() -> None:
    recorder, budget = ListRecorder(), RunBudget(max_steps=10, max_tokens=1000)
    error = ModelError("HTTP_503", "down", retryable=True)
    model = ScriptedModel(sequence(error, error, error))

    with pytest.raises(ModelError):
        call(model, recorder, budget, max_retries=2)
    assert len(recorder.events) == 3


def test_non_retryable_error_is_raised_at_once() -> None:
    recorder, budget = ListRecorder(), RunBudget(max_steps=10, max_tokens=1000)
    model = ScriptedModel(sequence(ModelError("HTTP_400", "bad request", retryable=False)))

    with pytest.raises(ModelError, match="bad request"):
        call(model, recorder, budget)
    assert len(recorder.events) == 1


def test_missing_usage_is_unknown_in_the_trail_but_counts_against_the_budget() -> None:
    recorder, budget = ListRecorder(), RunBudget(max_steps=5, max_tokens=1000)
    model = ScriptedModel(sequence(reply("x" * 400, usage=None)))

    call(model, recorder, budget)

    payload = recorder.events[0][2]
    assert payload["input_tokens"] is None and payload["output_tokens"] is None
    assert budget.tokens_used > 0


@pytest.mark.parametrize(
    ("budget", "limit"),
    [
        (RunBudget(max_steps=1, max_tokens=1000, steps_used=1), "max_steps"),
        (RunBudget(max_steps=9, max_tokens=100, tokens_used=100), "token_limit"),
    ],
)
def test_exhausted_budget_stops_before_calling(budget: RunBudget, limit: str) -> None:
    model = ScriptedModel(sequence(reply("never")))
    with pytest.raises(BudgetExceeded) as raised:
        call(model, ListRecorder(), budget)
    assert raised.value.limit == limit
    assert model.calls == []


# --- Gemini conversion -----------------------------------------------------------------


def test_tool_results_go_back_as_one_user_turn_after_the_model_turn() -> None:
    messages = [
        Message("user", "Where is my refund?"),
        Message(
            "assistant",
            tool_calls=[
                ToolCall("c1", "get_order", {"order_id": "ord_1"}),
                ToolCall("c2", "get_return_status", {"order_id": "ord_1"}),
            ],
        ),
        Message("tool", tool_call_id="c1", tool_name="get_order", tool_result={"success": True}),
        Message(
            "tool", tool_call_id="c2", tool_name="get_return_status", tool_result={"success": False}
        ),
    ]
    contents = to_contents(messages)

    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [p.function_call.name for p in contents[1].parts] == ["get_order", "get_return_status"]
    assert [p.function_response.id for p in contents[2].parts] == ["c1", "c2"]


def test_raw_model_turn_is_replayed_verbatim() -> None:
    raw = types.Content(role="model", parts=[types.Part(text="hi", thought_signature=b"sig")])
    assert to_contents([Message("assistant", "hi", raw=raw)])[0] is raw


def test_response_parsing_reads_calls_text_and_usage() -> None:
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="thinking", thought=True),
                        types.Part(text="Let me check."),
                        types.Part(
                            function_call=types.FunctionCall(
                                id="fc1", name="get_order", args={"order_id": "ord_1"}
                            )
                        ),
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=50, candidates_token_count=7, thoughts_token_count=3
        ),
        model_version="gemini-test-001",
    )

    parsed = from_response(response)

    assert parsed.text == "Let me check."
    assert parsed.tool_calls == [ToolCall("fc1", "get_order", {"order_id": "ord_1"})]
    assert parsed.usage == Usage(50, 7, None, 3)
    assert parsed.resolved_model == "gemini-test-001"
