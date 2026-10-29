"""Experiment plans: manifests under evals/plans/ and the quota estimate (W5.2, Q10, E§21).

A plan names a dataset, a split, the configurations to compare and the repetitions. Every
configuration runs the same task versions, in the same environment, with the same limits,
reviewer behaviour and fault configuration: those all come from the task, not the config, so
only the one thing a comparison changes (architecture, prompt or model) differs.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Connection, text

from evals.configs import named_config
from evals.db import EVAL_SCHEMA
from evals.experiments import ExperimentPlan

PLANS_DIR = Path(__file__).resolve().parent / "plans"
HEADROOM = 0.25  # retries, a longer-than-pilot run, an agent that wanders (E§21)


class PlanFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dataset: str
    split: str
    configs: list[str] = Field(min_length=1)
    repetitions: int = Field(default=1, gt=0)
    idempotency_key: str


def load_plan(path: str | Path, *, allow_test_split: bool = False) -> ExperimentPlan:
    file = Path(path)
    if not file.exists() and (PLANS_DIR / file).exists():
        file = PLANS_DIR / file
    if not file.exists() and (PLANS_DIR / f"{file}.yaml").exists():
        file = PLANS_DIR / f"{file}.yaml"
    plan = PlanFile.model_validate(yaml.safe_load(file.read_text(encoding="utf-8")))
    if len(set(plan.configs)) != len(plan.configs):
        raise ValueError(f"{plan.name}: a configuration is listed twice")
    return ExperimentPlan(
        name=plan.name,
        dataset=plan.dataset,
        split=plan.split,
        configs=[named_config(name) for name in plan.configs],
        repetitions=plan.repetitions,
        idempotency_key=plan.idempotency_key,
        allow_test_split=allow_test_split,
    )


def planned_runs(connection: Connection, plan: ExperimentPlan) -> int:
    from evals.loader import split_ids

    return len(split_ids(plan.dataset, plan.split)) * len(plan.configs) * plan.repetitions


def model_calls_per_run(connection: Connection, experiment_id: str) -> dict[str, Any]:
    """Mean model calls per finished attempt of an experiment (for sizing the next one)."""
    rows = connection.execute(
        text(
            "SELECT c.model_id, count(*) AS attempts, "
            "avg((a.usage->>'model_calls')::numeric) AS mean_calls "
            f"FROM {EVAL_SCHEMA}.attempts a JOIN {EVAL_SCHEMA}.runs r ON r.id = a.run_id "
            f"JOIN {EVAL_SCHEMA}.agent_configs c ON c.id = r.agent_config_id "
            "WHERE r.experiment_id = :id AND a.status = 'finished' GROUP BY c.model_id"
        ),
        {"id": experiment_id},
    )
    return {
        row.model_id: {"attempts": row.attempts, "mean_calls": float(row.mean_calls)}
        for row in rows
    }


def estimate_quota(
    connection: Connection, pilot_experiment_id: str, plan: ExperimentPlan, runs: int
) -> dict[str, Any]:
    """Requests needed = pilot mean model calls per run x planned runs, plus headroom."""
    measured = model_calls_per_run(connection, pilot_experiment_id)
    estimate: dict[str, Any] = {}
    per_model_runs: dict[str, int] = {}
    share = runs // max(len(plan.configs), 1)
    for row in plan.configs:
        per_model_runs[row["model_id"]] = per_model_runs.get(row["model_id"], 0) + share
    for model, count in per_model_runs.items():
        # A model the pilot did not use is estimated from the pilot's overall mean.
        pilot = measured.get(model) or {
            "mean_calls": sum(m["mean_calls"] for m in measured.values()) / max(len(measured), 1),
            "attempts": 0,
        }
        needed = pilot["mean_calls"] * count
        estimate[model] = {
            "planned_runs": count,
            "pilot_mean_calls_per_run": pilot["mean_calls"],
            "pilot_runs": pilot["attempts"],
            "requests": round(needed),
            "requests_with_headroom": round(needed * (1 + HEADROOM)),
        }
    return estimate


def quota_hits(connection: Connection) -> list[dict[str, Any]]:
    """Quota errors seen in eval traces, by the quota the provider named (first hit first)."""
    rows = connection.execute(
        text(
            "SELECT payload->>'error_detail' AS quota, payload->>'model_id' AS model_id, "
            "count(*) AS hits, min(occurred_at) AS first_seen "
            f"FROM {EVAL_SCHEMA}.trace_events WHERE event_type = 'model_call' "
            "AND payload->>'outcome' = 'error' AND payload->>'error_code' = 'HTTP_429' "
            "GROUP BY 1, 2 ORDER BY first_seen"
        )
    )
    return [dict(row._mapping) for row in rows]
