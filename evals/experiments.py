"""Create an experiment: its manifest, runs and one `eval_attempt` job per run (W4.8, E§18).

Everything is written in one transaction, so an experiment never exists without its runs and
jobs. A repeated create with the same idempotency key returns the existing experiment.
Configurations are interleaved in run order (E§22): each task's runs rotate the starting
configuration, so quota or time-of-day effects do not favour one.
"""

import hashlib
import json
import secrets
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.jobs import queue
from evals.configs import register_agent_config, sha256
from evals.db import EVAL_SCHEMA
from evals.evaluators.suite import EVALUATORS
from evals.loader import LoadedTask, load_dataset, split_ids

JOB_TYPE = "eval_attempt"
ENVIRONMENT_VERSION = "1"  # evals/environment.py: reset, fixture loading, keyed faults
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentPlan:
    name: str
    dataset: str
    split: str
    configs: list[dict[str, Any]]  # agent_configs rows
    repetitions: int = 1
    idempotency_key: str | None = None


def create_experiment(db: Session, plan: ExperimentPlan) -> tuple[str, bool]:
    """Returns (experiment ID, created). Commits."""
    key = plan.idempotency_key or f"{plan.name}:{secrets.token_hex(4)}"
    connection = db.connection()
    existing = connection.execute(
        text(f"SELECT id FROM {EVAL_SCHEMA}.experiments WHERE idempotency_key = :key"),
        {"key": key},
    ).scalar_one_or_none()
    if existing is not None:
        return existing, False

    tasks = {task.id: task for task in load_dataset(plan.dataset)}
    wanted = split_ids(plan.dataset, plan.split)
    dataset_version = register_dataset(connection, plan.dataset, tasks)
    task_versions = {task_id: register_task(connection, tasks[task_id]) for task_id in wanted}
    for row in plan.configs:
        register_agent_config(connection, row)

    experiment_id = f"exp_{secrets.token_hex(8)}"
    manifest = {
        "dataset_version": dataset_version,
        "split": plan.split,
        "tasks": {t: tasks[t].evaluator.content_hash for t in wanted},
        "configs": {
            row["id"]: {"config_hash": row["config_hash"], "model_id": row["model_id"]}
            for row in plan.configs
        },
        "repetitions": plan.repetitions,
        "limits": {t: tasks[t].evaluator.evaluation.limits.model_dump() for t in wanted},
        "environment_version": ENVIRONMENT_VERSION,
        "evaluator_versions": {name: version for name, version, _ in EVALUATORS},
        "code_commit": code_commit(),
        "lockfile_hash": file_hash(ROOT / "uv.lock"),
    }
    connection.execute(
        text(
            f"INSERT INTO {EVAL_SCHEMA}.experiments (id, name, idempotency_key, "
            "dataset_version_id, split, repetitions, manifest) VALUES (:id, :name, :key, "
            ":dataset_version, :split, :repetitions, CAST(:manifest AS jsonb))"
        ),
        {
            "id": experiment_id,
            "name": plan.name,
            "key": key,
            "dataset_version": dataset_version["id"],
            "split": plan.split,
            "repetitions": plan.repetitions,
            "manifest": json.dumps(manifest),
        },
    )
    # Distinct, increasing availability times keep the claim order exactly the run order. They
    # start a minute back, so a small clock difference with the database never delays a job.
    start = datetime.now(UTC) - timedelta(minutes=1)
    for index, (task_id, repetition, config_id) in enumerate(
        run_order(wanted, [row["id"] for row in plan.configs], plan.repetitions)
    ):
        run_id = f"erun_{secrets.token_hex(8)}"
        connection.execute(
            text(
                f"INSERT INTO {EVAL_SCHEMA}.runs (id, experiment_id, agent_config_id, "
                "task_version_id, repetition) VALUES (:id, :experiment, :config, :task, :rep)"
            ),
            {
                "id": run_id,
                "experiment": experiment_id,
                "config": config_id,
                "task": task_versions[task_id],
                "rep": repetition,
            },
        )
        queue.enqueue(
            db,
            JOB_TYPE,
            {"run_id": run_id},
            dedupe_key=f"eval:{run_id}",
            available_at=start + timedelta(microseconds=index),
        )
    db.commit()
    return experiment_id, True


def run_order(task_ids: list[str], config_ids: list[str], repetitions: int):
    """(task, repetition, config) with configurations rotated per task and repetition."""
    position = 0
    for task_id in task_ids:
        for repetition in range(1, repetitions + 1):
            for offset in range(len(config_ids)):
                yield task_id, repetition, config_ids[(position + offset) % len(config_ids)]
            position += 1


def register_dataset(
    connection: Connection, dataset: str, tasks: dict[str, LoadedTask]
) -> dict[str, Any]:
    name, _, version = dataset.rpartition("-v")
    manifest = {
        t.id: {"version": t.evaluator.version, "content_hash": t.evaluator.content_hash}
        for t in tasks.values()
    }
    splits = {split: split_ids(dataset, split) for split in ("dev", "validation", "test", "pilot")}
    content_hash = sha256({"tasks": manifest, "splits": splits})
    stored = connection.execute(
        text(f"SELECT content_hash FROM {EVAL_SCHEMA}.dataset_versions WHERE id = :id"),
        {"id": dataset},
    ).scalar_one_or_none()
    if stored is None:
        connection.execute(
            text(f"INSERT INTO {EVAL_SCHEMA}.datasets (id) VALUES (:id) ON CONFLICT DO NOTHING"),
            {"id": name},
        )
        connection.execute(
            text(
                f"INSERT INTO {EVAL_SCHEMA}.dataset_versions (id, dataset_id, version, "
                "content_hash, manifest, splits) VALUES (:id, :dataset, :version, :hash, "
                "CAST(:manifest AS jsonb), CAST(:splits AS jsonb))"
            ),
            {
                "id": dataset,
                "dataset": name,
                "version": int(version),
                "hash": content_hash,
                "manifest": json.dumps(manifest),
                "splits": json.dumps(splits),
            },
        )
    elif stored != content_hash:
        raise ValueError(
            f"{dataset} changed since it was registered; release it as a new dataset version"
        )
    return {"id": dataset, "content_hash": content_hash}


def register_task(connection: Connection, task: LoadedTask) -> str:
    view = task.evaluator
    task_version_id = f"{view.task_id}@v{view.version}"
    stored = connection.execute(
        text(f"SELECT content_hash FROM {EVAL_SCHEMA}.task_versions WHERE id = :id"),
        {"id": task_version_id},
    ).scalar_one_or_none()
    if stored is None:
        connection.execute(
            text(
                f"INSERT INTO {EVAL_SCHEMA}.task_versions (id, task_id, version, content_hash, "
                "category, difficulty, scenario_family, agent_view, evaluator_view) VALUES "
                "(:id, :task, :version, :hash, :category, :difficulty, :family, "
                "CAST(:agent AS jsonb), CAST(:evaluator AS jsonb))"
            ),
            {
                "id": task_version_id,
                "task": view.task_id,
                "version": view.version,
                "hash": view.content_hash,
                "category": view.category,
                "difficulty": view.difficulty,
                "family": view.scenario_family,
                "agent": task.agent.model_dump_json(),
                "evaluator": view.model_dump_json(),
            },
        )
    elif stored != view.content_hash:
        raise ValueError(f"{task_version_id} is registered with different content")
    return task_version_id


def code_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
