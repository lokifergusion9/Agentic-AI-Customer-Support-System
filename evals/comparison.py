"""Compare two configurations on the same tasks (W5.3, E§22, E§25).

Only **compatible** experiments are compared: same dataset version, split, environment and
evaluator versions, and per-task limits. Anything else would make a difference between the two
configurations impossible to attribute to the configuration.

The result names the tasks that improved or regressed. A task is compared on its mean success
over the runs of each side (one run, or the repetitions), so a flaky task shows as a smaller
change rather than a coin flip. A task one side could not score is `not_comparable`, listed
apart: it neither helps nor hurts either side.
"""

import math
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection, text

from evals import metrics
from evals.db import EVAL_SCHEMA
from evals.metrics import RunData, load_prices, load_runs

COMPATIBILITY_FIELDS = ("dataset_version", "split", "environment_version", "evaluator_versions")


class IncompatibleExperiments(ValueError):
    def __init__(self, differences: list[str]) -> None:
        super().__init__("experiments are not compatible: " + "; ".join(differences))
        self.differences = differences


@dataclass(frozen=True)
class Side:
    experiment_id: str
    config_id: str | None = None  # needed when the experiment has more than one configuration


def load_manifest(connection: Connection, experiment_id: str) -> dict[str, Any]:
    row = connection.execute(
        text(
            f"SELECT manifest, dataset_version_id, split FROM {EVAL_SCHEMA}.experiments "
            "WHERE id = :id"
        ),
        {"id": experiment_id},
    ).one_or_none()
    if row is None:
        raise LookupError(f"no experiment {experiment_id!r}")
    return {**row.manifest, "dataset_version_id": row.dataset_version_id, "split": row.split}


def differences(a: dict[str, Any], b: dict[str, Any], tasks: set[str]) -> list[str]:
    """Why two manifests cannot be compared, for the tasks both sides ran."""
    found = []
    for name in COMPATIBILITY_FIELDS:
        if a.get(name) != b.get(name):
            found.append(f"{name}: {a.get(name)!r} vs {b.get(name)!r}")
    for task in sorted(tasks):
        if a["limits"].get(task) != b["limits"].get(task):
            found.append(f"limits differ for {task}")
        if a["tasks"].get(task) != b["tasks"].get(task):
            found.append(f"task {task} is a different version")
    return found


def sign_test_p(improved: int, regressed: int) -> float | None:
    """Two-sided exact binomial p-value for improved vs regressed tasks (ties are dropped).

    A rough check, not a verdict: with few differing tasks no result is significant.
    """
    n = improved + regressed
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(connection: Connection, baseline: Side, candidate: Side) -> dict[str, Any]:
    manifests = {
        s.experiment_id: load_manifest(connection, s.experiment_id) for s in (baseline, candidate)
    }
    runs_a = select_runs(connection, baseline)
    runs_b = select_runs(connection, candidate)
    tasks_a, tasks_b = {r.task_id for r in runs_a}, {r.task_id for r in runs_b}
    common = tasks_a & tasks_b
    problems = differences(
        manifests[baseline.experiment_id], manifests[candidate.experiment_id], common
    )
    if not common:
        problems.append("the two sides share no task")
    if problems:
        raise IncompatibleExperiments(problems)

    prices = load_prices(connection)
    shared_a = [r for r in runs_a if r.task_id in common]
    shared_b = [r for r in runs_b if r.task_id in common]
    tasks = [task_change(task, shared_a, shared_b) for task in sorted(common)]
    improved = [t["task_id"] for t in tasks if t["change"] == "improved"]
    regressed = [t["task_id"] for t in tasks if t["change"] == "regressed"]
    card_a, card_b = metrics.card(shared_a, prices), metrics.card(shared_b, prices)
    return {
        "compatible": True,
        "baseline": describe(baseline, runs_a),
        "candidate": describe(candidate, runs_b),
        "tasks_compared": len(common),
        "tasks_only_in_baseline": sorted(tasks_a - common),
        "tasks_only_in_candidate": sorted(tasks_b - common),
        "summary": {
            "improved": len(improved),
            "regressed": len(regressed),
            "unchanged": sum(t["change"] in ("both_pass", "both_fail", "tied") for t in tasks),
            "not_comparable": sum(t["change"] == "not_comparable" for t in tasks),
            "sign_test_p": sign_test_p(len(improved), len(regressed)),
        },
        "improved_tasks": improved,
        "regressed_tasks": regressed,
        "tasks": tasks,
        "differences": differences_of_cards(card_a, card_b),
        "by_category": by_category(shared_a, shared_b, prices),
        "scorecards": {"baseline": card_a, "candidate": card_b},
    }


def select_runs(connection: Connection, side: Side) -> list[RunData]:
    runs = load_runs(connection, side.experiment_id)
    configs = sorted({r.config_id for r in runs})
    if side.config_id is None:
        if len(configs) != 1:
            raise ValueError(
                f"experiment {side.experiment_id} has {len(configs)} configurations "
                f"({', '.join(configs)}): name the one to compare"
            )
        return runs
    if side.config_id not in configs:
        raise ValueError(f"experiment {side.experiment_id} has no configuration {side.config_id!r}")
    return [r for r in runs if r.config_id == side.config_id]


def describe(side: Side, runs: list[RunData]) -> dict[str, Any]:
    return {
        "experiment_id": side.experiment_id,
        "config_id": runs[0].config_id if runs else side.config_id,
        "model_id": runs[0].model_id if runs else None,
        "runs": len(runs),
    }


def task_change(task_id: str, runs_a: list[RunData], runs_b: list[RunData]) -> dict[str, Any]:
    a = [r for r in runs_a if r.task_id == task_id]
    b = [r for r in runs_b if r.task_id == task_id]
    mean_a, mean_b = mean_success(a), mean_success(b)
    if mean_a is None or mean_b is None:
        change = "not_comparable"
    elif mean_b > mean_a:
        change = "improved"
    elif mean_b < mean_a:
        change = "regressed"
    else:
        change = "both_pass" if mean_a == 1 else "both_fail" if mean_a == 0 else "tied"
    return {
        "task_id": task_id,
        "category": (a or b)[0].category,
        "change": change,
        "baseline": side_of_task(a, mean_a),
        "candidate": side_of_task(b, mean_b),
    }


def mean_success(runs: list[RunData]) -> float | None:
    scorable = [r for r in runs if r.scorable]
    return sum(r.success for r in scorable) / len(scorable) if scorable else None


def side_of_task(runs: list[RunData], mean: float | None) -> dict[str, Any]:
    failures = sorted(
        {
            f"{s['evaluator_name']}.{s['dimension']}:{s['reason_code']}"
            for r in runs
            for s in r.scores
            if s["status"] == "fail"
        }
    )
    return {
        "runs": len(runs),
        "scorable_runs": sum(r.scorable for r in runs),
        "mean_success": mean,
        "run_ids": [r.run_id for r in runs],
        "failures": failures,
    }


def delta(a: float | None, b: float | None) -> dict[str, Any]:
    return {"baseline": a, "candidate": b, "difference": None if a is None or b is None else b - a}


def differences_of_cards(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def rate(card: dict[str, Any], key: str) -> float | None:
        return card[key]["value"]

    def latency(card: dict[str, Any], key: str) -> float | None:
        return card["latency"]["active_ms"][key]

    usage_a, usage_b = a["tokens_and_cost"], b["tokens_and_cost"]
    return {
        "verified_success_rate": delta(
            rate(a, "verified_success_rate"), rate(b, "verified_success_rate")
        ),
        "strict_success_rate": delta(
            rate(a, "strict_success_rate"), rate(b, "strict_success_rate")
        ),
        "evaluation_coverage": delta(
            rate(a, "evaluation_coverage"), rate(b, "evaluation_coverage")
        ),
        "executed_violation_rate": delta(
            rate(a, "executed_violation_rate"), rate(b, "executed_violation_rate")
        ),
        "tool_calls_per_run": delta(rate(a, "tool_calls_per_run"), rate(b, "tool_calls_per_run")),
        "model_calls_per_run": delta(
            rate(usage_a, "model_calls_per_run"), rate(usage_b, "model_calls_per_run")
        ),
        "tokens_per_run": delta(rate(usage_a, "tokens_per_run"), rate(usage_b, "tokens_per_run")),
        "estimated_cost_per_successful_run_usd": delta(
            usage_a["estimated_cost_per_successful_run_usd"],
            usage_b["estimated_cost_per_successful_run_usd"],
        ),
        "active_ms_p50": delta(latency(a, "p50"), latency(b, "p50")),
        "active_ms_p95": delta(latency(a, "p95"), latency(b, "p95")),
    }


def by_category(
    runs_a: list[RunData], runs_b: list[RunData], prices: list[dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for category in sorted({r.category for r in runs_a + runs_b}):
        a = metrics.card([r for r in runs_a if r.category == category], prices)
        b = metrics.card([r for r in runs_b if r.category == category], prices)
        result[category] = {
            "verified_success_rate": delta(
                a["verified_success_rate"]["value"], b["verified_success_rate"]["value"]
            ),
            "evaluation_coverage": delta(
                a["evaluation_coverage"]["value"], b["evaluation_coverage"]["value"]
            ),
        }
    return result
