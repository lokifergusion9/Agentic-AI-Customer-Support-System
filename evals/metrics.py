"""Metrics with explicit denominators (W5.1, E§20, §15).

Computed per configuration, and per category, from `eval.scores`, `eval.trace_events` and
`eval.attempts`. Every rate comes with its numerator and denominator, and a rate over an empty
denominator is None (unknown), never 0. A run's result is its **final finished attempt**: an
interrupted attempt or an infrastructure retry is not counted a second time.

- planned runs: every run of the experiment, whatever happened to it.
- scorable runs: finished with a final attempt and a pass/fail/partial evaluation. A run that
  hit an infrastructure error, was cancelled, or whose evaluators raised, is not scorable.
- successful runs: scorable, with verified success (E§20).

The scorecard has several numbers on purpose, not one total: a configuration with a high
success rate over a low coverage is not the same as one that finished the whole experiment.
"""

import random
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from sqlalchemy import Connection, text

from evals.db import EVAL_SCHEMA
from evals.evaluators.base import pair_tool_calls

SCORED = ("pass", "fail", "partial")
INVALID_CALLS = ("INVALID_ARGUMENTS", "TOOL_NOT_ALLOWED")
COST_LABEL = "estimate at paid rates; the free tier itself costs $0 (Q4)"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261029


@dataclass
class RunData:
    run_id: str
    config_id: str
    model_id: str
    task_id: str
    task_version_id: str
    category: str
    repetition: int
    execution_status: str
    evaluation_status: str
    created_at: datetime
    task: dict[str, Any]  # the evaluator view
    first_started_at: datetime | None = None
    final: dict[str, Any] | None = None  # the final finished attempt
    scores: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scorable(self) -> bool:
        return self.final is not None and self.evaluation_status in SCORED

    @property
    def success(self) -> bool:
        return self.scorable and self.final["verified_success"] is True  # type: ignore[index]

    @property
    def human_assisted(self) -> bool:
        """The run asked for approval, which the simulator decided (E§8)."""
        return any(e["event_type"] == "proposal_created" for e in self.events)

    def score_rows(self, evaluator: str, dimension: str | None = None) -> list[dict[str, Any]]:
        return [
            s
            for s in self.scores
            if s["evaluator_name"] == evaluator
            and (dimension is None or s["dimension"] == dimension)
        ]


# --- Loading -----------------------------------------------------------------------------------


def load_runs(connection: Connection, experiment_id: str) -> list[RunData]:
    rows = connection.execute(
        text(
            "SELECT r.id AS run_id, r.agent_config_id AS config_id, c.model_id, t.task_id, "
            "r.task_version_id, t.category, r.repetition, r.execution_status, "
            "r.evaluation_status, r.created_at, t.evaluator_view AS task "
            f"FROM {EVAL_SCHEMA}.runs r "
            f"JOIN {EVAL_SCHEMA}.task_versions t ON t.id = r.task_version_id "
            f"JOIN {EVAL_SCHEMA}.agent_configs c ON c.id = r.agent_config_id "
            "WHERE r.experiment_id = :id ORDER BY r.created_at, r.id"
        ),
        {"id": experiment_id},
    ).all()
    runs = {row.run_id: RunData(**row._mapping) for row in rows}

    attempts = connection.execute(
        text(
            f"SELECT a.* FROM {EVAL_SCHEMA}.attempts a JOIN {EVAL_SCHEMA}.runs r ON r.id = a.run_id "
            "WHERE r.experiment_id = :id ORDER BY a.run_id, a.attempt_number"
        ),
        {"id": experiment_id},
    ).all()
    by_attempt: dict[str, RunData] = {}
    for attempt in attempts:
        run = runs[attempt.run_id]
        if run.first_started_at is None:
            run.first_started_at = attempt.started_at
        if attempt.status == "finished":
            run.final = dict(attempt._mapping)  # the latest finished attempt wins
    for run in runs.values():
        if run.final is not None:
            by_attempt[run.final["id"]] = run

    if by_attempt:
        ids = list(by_attempt)
        for score in connection.execute(
            text(f"SELECT * FROM {EVAL_SCHEMA}.scores WHERE attempt_id = ANY(:ids) ORDER BY id"),
            {"ids": ids},
        ):
            by_attempt[score.attempt_id].scores.append(dict(score._mapping))
        for event in connection.execute(
            text(
                f"SELECT attempt_id, sequence_number, actor, event_type, payload, occurred_at "
                f"FROM {EVAL_SCHEMA}.trace_events WHERE attempt_id = ANY(:ids) "
                "ORDER BY attempt_id, sequence_number"
            ),
            {"ids": ids},
        ):
            by_attempt[event.attempt_id].events.append(dict(event._mapping))
    return list(runs.values())


def load_prices(connection: Connection) -> list[dict[str, Any]]:
    rows = connection.execute(text(f"SELECT * FROM {EVAL_SCHEMA}.price_versions"))
    return [dict(row._mapping) for row in rows]


# --- Helpers -----------------------------------------------------------------------------------


def ratio(numerator: float, denominator: float) -> dict[str, Any]:
    """A rate with its parts; the rate is None when the denominator is 0."""
    return {
        "value": None if denominator == 0 else numerator / denominator,
        "n": numerator,
        "of": denominator,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def agent_calls(run: RunData):
    """Tool calls the model attempted (not the workflow's own steps)."""
    return [c for c in pair_tool_calls(run.events) if c.actor.startswith("agent:")]


# --- One card ----------------------------------------------------------------------------------


def card(runs: list[RunData], prices: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The scorecard for a set of runs (one configuration, or one category of it)."""
    scorable = [r for r in runs if r.scorable]
    successful = [r for r in scorable if r.success]
    result: dict[str, Any] = {
        "planned_runs": len(runs),
        "scorable_runs": len(scorable),
        "successful_runs": len(successful),
        "verified_success_rate": ratio(len(successful), len(scorable)),
        "strict_success_rate": ratio(len(successful), len(runs)),
        "evaluation_coverage": ratio(len(scorable), len(runs)),
        "run_outcomes": count_by(runs, lambda r: f"{r.execution_status}/{r.evaluation_status}"),
    }
    result.update(policy_metrics(scorable))
    result.update(tool_metrics(scorable))
    result["policy_retrieval"] = retrieval_metrics(scorable)
    result["latency"] = latency_metrics(runs)
    result["resolution"] = {
        "human_assisted": success_split([r for r in scorable if r.human_assisted]),
        "fully_automated": success_split([r for r in scorable if not r.human_assisted]),
    }
    result["tokens_and_cost"] = usage_metrics(scorable, successful, prices or [])
    result["repetitions"] = repetition_metrics(runs)
    result["failure_reasons"] = failure_reasons(scorable)
    return result


def count_by(runs: list[RunData], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[key(run)] = counts.get(key(run), 0) + 1
    return dict(sorted(counts.items()))


def success_split(runs: list[RunData]) -> dict[str, Any]:
    return {
        "runs": len(runs),
        "successful": sum(r.success for r in runs),
        "success_rate": ratio(sum(r.success for r in runs), len(runs)),
    }


def policy_metrics(scorable: list[RunData]) -> dict[str, Any]:
    """Unsafe attempts (blocked by the tools) and executed violations, kept apart (E§12)."""
    unsafe = hard = duplicates = 0
    for run in scorable:
        policy = [s for s in run.score_rows("policy") if s["status"] == "fail"]
        unsafe += any(s["evidence"].get("severity") == "attempt" for s in policy)
        hard += any(s["evidence"].get("severity") == "hard" for s in policy)
        for s in run.score_rows("outcome", "financial_effects"):
            if s["reason_code"] == "DUPLICATE_FINANCIAL_EFFECT":
                duplicates += s["evidence"]["observed_count"] - s["evidence"]["expected_count"]
    applicable = sum(bool([s for s in r.score_rows("policy") if s["applicable"]]) for r in scorable)
    return {
        "unsafe_attempt_rate": ratio(unsafe, applicable),
        "executed_violation_rate": ratio(hard, applicable),
        "duplicate_refund_count": duplicates,
    }


def tool_metrics(scorable: list[RunData]) -> dict[str, Any]:
    attempted = invalid = correct = 0
    required = satisfied = 0
    for run in scorable:
        allowed = set(run.task["evaluation"]["allowed_actions"])
        seen: set[tuple[str, str]] = set()
        for call in agent_calls(run):
            attempted += 1
            invalid += call.error_code in INVALID_CALLS
            fingerprint = (call.tool, repr(sorted(call.args.items())))
            repeat = fingerprint in seen
            if call.outcome == "ok":
                seen.add(fingerprint)
                correct += call.tool in allowed and not repeat
        for s in run.score_rows("trajectory", "required_steps"):
            if s["applicable"]:
                required += len(s["evidence"]["required"])
                satisfied += len(s["evidence"]["required"]) - len(s["evidence"]["missing"])
    return {
        "tool_calls_per_run": ratio(attempted, len(scorable)),
        "tool_schema_validity": ratio(attempted - invalid, attempted),
        "tool_call_precision": ratio(correct, attempted),
        "required_call_recall": ratio(satisfied, required),
    }


def retrieval_metrics(scorable: list[RunData]) -> dict[str, Any]:
    """Did `search_policy` return the policy section the task relies on? (§15)

    `any_required_section`: at least one required section was retrieved.
    `all_required_sections`: every required section was retrieved.
    """
    needing = [r for r in scorable if r.task["evaluation"]["applicable_policy"]]
    any_hit = full_hit = 0
    for run in needing:
        required = set(run.task["evaluation"]["applicable_policy"])
        retrieved = {
            citation
            for e in run.events
            if e["event_type"] == "tool_finished" and e["payload"].get("tool") == "search_policy"
            for citation in e["payload"].get("citations", [])
        }
        any_hit += bool(required & retrieved)
        full_hit += required <= retrieved
    return {
        "any_required_section": ratio(any_hit, len(needing)),
        "all_required_sections": ratio(full_hit, len(needing)),
    }


def latency_metrics(runs: list[RunData]) -> dict[str, Any]:
    """Queue time, active execution and reviewer wait, reported separately (§15)."""
    queue, active, wait = [], [], []
    for run in runs:
        if run.first_started_at is not None:
            queue.append((run.first_started_at - run.created_at).total_seconds() * 1000)
        if run.final is not None:
            if run.final["active_ms"] is not None:
                active.append(run.final["active_ms"])
            if run.final["approval_wait_ms"] is not None:
                wait.append(run.final["approval_wait_ms"])
    return {
        "queue_ms": distribution(queue),
        "active_ms": distribution(active),
        "approval_wait_ms": distribution(wait),
    }


def failure_reasons(scorable: list[RunData]) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for run in scorable:
        for s in run.scores:
            if s["status"] == "fail":
                key = f"{s['evaluator_name']}.{s['dimension']}:{s['reason_code']}"
                reasons[key] = reasons.get(key, 0) + 1
    return dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0])))


# --- Tokens and cost ---------------------------------------------------------------------------


def run_tokens(run: RunData) -> dict[str, int | None]:
    usage = (run.final or {}).get("usage") or {}
    known = usage.get("input_tokens") is not None and usage.get("output_tokens") is not None
    reasoning = usage.get("reasoning_tokens") or 0
    return {
        "input": usage.get("input_tokens"),
        "cached": usage.get("cached_input_tokens") or 0,
        "output": None
        if usage.get("output_tokens") is None
        else usage["output_tokens"] + reasoning,
        "total": usage["input_tokens"] + usage["output_tokens"] + reasoning if known else None,
    }


def price_for(
    prices: list[dict[str, Any]], model_id: str, on: date | None
) -> dict[str, Any] | None:
    candidates = [
        p for p in prices if p["model_id"] == model_id and (on is None or p["effective_from"] <= on)
    ]
    return max(candidates, key=lambda p: p["effective_from"], default=None)


def run_cost(run: RunData, prices: list[dict[str, Any]]) -> float | None:
    """USD at paid rates, or None (unknown, never free) without usage or a registered price."""
    tokens = run_tokens(run)
    if tokens["total"] is None:
        return None
    finished = run.final["finished_at"].date() if run.final and run.final["finished_at"] else None
    price = price_for(prices, run.model_id, finished)
    if price is None:
        return None
    cached = tokens["cached"]
    cached_rate = price["cached_input_per_mtok_usd"] or price["input_per_mtok_usd"]
    return (
        float(
            (tokens["input"] - cached) * price["input_per_mtok_usd"]
            + cached * cached_rate
            + tokens["output"] * price["output_per_mtok_usd"]
        )
        / 1_000_000
    )


def usage_metrics(
    scorable: list[RunData], successful: list[RunData], prices: list[dict[str, Any]]
) -> dict[str, Any]:
    tokens = [run_tokens(r) for r in scorable]
    unknown_tokens = sum(t["total"] is None for t in tokens)
    total_tokens = sum(t["total"] for t in tokens if t["total"] is not None)
    costs = [run_cost(r, prices) for r in scorable]
    unknown_cost = sum(c is None for c in costs)
    total_cost = sum(c for c in costs if c is not None)
    complete = bool(scorable) and unknown_cost == 0
    calls = [((r.final or {}).get("usage") or {}).get("model_calls") for r in scorable]
    return {
        "model_calls_per_run": ratio(sum(c for c in calls if c is not None), len(scorable)),
        "tokens_per_run": ratio(total_tokens, len(scorable) - unknown_tokens),
        "runs_with_unknown_usage": unknown_tokens,
        "total_tokens": total_tokens,
        "total_tokens_complete": unknown_tokens == 0,
        "tokens_per_successful_run": (
            total_tokens / len(successful) if successful and unknown_tokens == 0 else None
        ),
        "cost_label": COST_LABEL,
        "estimated_cost_usd": total_cost if costs else None,
        "runs_with_unknown_cost": unknown_cost,
        "estimated_cost_per_successful_run_usd": (
            total_cost / len(successful) if successful and complete else None
        ),
    }


# --- Repetitions -------------------------------------------------------------------------------


def repetition_metrics(runs: list[RunData]) -> dict[str, Any]:
    """Success across repetitions, per task, with uncertainty resampled over tasks (E§22).

    Repetitions of one task stay together: the unit of resampling is the task, so a task run
    three times is not counted as three independent tasks.
    """
    per_task: dict[str, list[bool]] = {}
    for run in runs:
        if run.scorable:
            per_task.setdefault(run.task_id, []).append(run.success)
    means = {task: sum(results) / len(results) for task, results in per_task.items()}
    summary = {
        task: {"runs": len(r), "successes": sum(r), "mean": means[task]}
        for task, r in sorted(per_task.items())
    }
    repeated = [t for t, r in per_task.items() if len(r) > 1]
    return {
        "tasks": len(per_task),
        "mean_task_success": statistics.fmean(means.values()) if means else None,
        "ci95": bootstrap_ci(list(means.values())),
        "passed_all_repetitions": ratio(sum(all(per_task[t]) for t in repeated), len(repeated)),
        "flaky_tasks": sorted(t for t, m in means.items() if 0 < m < 1),
        "per_task": summary,
    }


def bootstrap_ci(task_means: list[float]) -> list[float] | None:
    """95% percentile interval of the mean task success, resampling tasks with replacement."""
    if len(task_means) < 2:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(task_means)
    samples = sorted(
        statistics.fmean(rng.choices(task_means, k=n)) for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return [
        samples[int(0.025 * BOOTSTRAP_RESAMPLES)],
        samples[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
    ]


# --- Scorecards for an experiment --------------------------------------------------------------


def scorecards(runs: list[RunData], prices: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """{config_id: {overall: card, by_category: {category: card}}}."""
    result: dict[str, Any] = {}
    for config_id in sorted({r.config_id for r in runs}):
        mine = [r for r in runs if r.config_id == config_id]
        result[config_id] = {
            "overall": card(mine, prices),
            "by_category": {
                category: card([r for r in mine if r.category == category], prices)
                for category in sorted({r.category for r in mine})
            },
        }
    return result


def experiment_scorecard(connection: Connection, experiment_id: str) -> dict[str, Any]:
    return scorecards(load_runs(connection, experiment_id), load_prices(connection))
