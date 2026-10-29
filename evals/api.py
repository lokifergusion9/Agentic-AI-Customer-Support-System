"""The evaluation API (W5.4, E§24, Q12): `uv run uvicorn evals.api:app --port 8100`.

Every endpoint is under `/eval` and needs a JWT with the `operator` role: a customer or
reviewer token is refused. It is a separate application from the agent's API, so the agent's
`/runs` endpoints cannot collide with it and nothing under `app/` ever imports `evals/`. It
creates experiments and reads results; the runs themselves are executed by the eval worker
(`python -m evals.run_evals work`).
"""

import io
import re
import zipfile
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.agents.config import AgentConfig
from app.auth.dependencies import get_token_claims
from app.auth.tokens import TokenClaims
from app.db.session import get_db
from evals import metrics
from evals.comparison import IncompatibleExperiments, Side, compare
from evals.configs import LIMIT_FIELDS, config_row, named_config, register_agent_config
from evals.db import EVAL_SCHEMA
from evals.evaluators.base import AttemptRecord
from evals.experiments import ExperimentPlan, create_experiment, register_dataset, register_task
from evals.loader import TaskVersionError, load_dataset
from evals.reports import export_experiment
from evals.task_format import AgentInput, EvaluatorView
from evals.web import pages as web_pages

DATASET = (
    r"^[a-z0-9]+(-[a-z0-9]+)*-v[0-9]+$"  # a directory name under evals/datasets/, nothing else
)
SPLIT = r"^[a-z]+$"


def require_operator(claims: Annotated[TokenClaims, Depends(get_token_claims)]) -> TokenClaims:
    if claims.role != "operator":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Operator role required")
    return claims


Db = Annotated[Session, Depends(get_db)]
router = APIRouter(prefix="/eval", tags=["evaluation"], dependencies=[Depends(require_operator)])


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Datasets ----------------------------------------------------------------------------------


class ImportDataset(Strict):
    dataset: str = Field(pattern=DATASET, description="e.g. support-v1")


@router.post("/datasets/import")
def import_dataset(body: ImportDataset, db: Db) -> dict[str, Any]:
    """Validate a dataset version against the task format and its lock, and store it."""
    try:
        tasks = {task.id: task for task in load_dataset(body.dataset)}
    except (TaskVersionError, ValidationError, ValueError) as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from None
    if not tasks:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no dataset {body.dataset!r}")
    connection = db.connection()
    version = register_dataset(connection, body.dataset, tasks)
    for task in tasks.values():
        register_task(connection, task)
    db.commit()
    return {
        "dataset_version": version["id"],
        "content_hash": version["content_hash"],
        "tasks": len(tasks),
    }


@router.get("/datasets")
def list_datasets(db: Db) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            "SELECT id, dataset_id, version, content_hash, created_at, "
            "(SELECT count(*) FROM jsonb_object_keys(manifest)) AS tasks, splits "
            f"FROM {EVAL_SCHEMA}.dataset_versions ORDER BY dataset_id, version"
        )
    )
    return [{**row._mapping, "splits": {k: len(v) for k, v in row.splits.items()}} for row in rows]


# --- Agent configurations ----------------------------------------------------------------------


class ConfigIn(Strict):
    name: str | None = Field(default=None, description="a registered name, e.g. lite_single_v1")
    id: str | None = None
    provider: str = "gemini"
    model_id: str | None = None
    prompt_version: str | None = None
    architecture: str | None = None
    limits: dict[str, float] = {}


@router.post("/agent-configs")
def register_config(body: ConfigIn, db: Db, response: Response) -> dict[str, Any]:
    """Register an immutable configuration: a known name, or the full set of settings."""
    try:
        if body.name:
            row = named_config(body.name)
        else:
            if not (body.id and body.model_id and body.prompt_version and body.architecture):
                raise ValueError("give a name, or id, model_id, prompt_version and architecture")
            unknown = set(body.limits) - set(LIMIT_FIELDS)
            if unknown:
                raise ValueError(f"unknown limits: {sorted(unknown)}")
            config = AgentConfig(
                provider=body.provider,
                model_id=body.model_id,
                prompt_version=body.prompt_version,
                architecture=body.architecture,  # type: ignore[arg-type]
                **body.limits,  # type: ignore[arg-type]
            )
            row = config_row(body.id, config)
        existed = db.execute(
            text(f"SELECT 1 FROM {EVAL_SCHEMA}.agent_configs WHERE id = :id"), {"id": row["id"]}
        ).first()
        register_agent_config(db.connection(), row)
    except ValueError as error:
        conflict = "immutable" in str(error)
        raise HTTPException(
            status.HTTP_409_CONFLICT if conflict else status.HTTP_422_UNPROCESSABLE_CONTENT,
            str(error),
        ) from None
    db.commit()
    response.status_code = status.HTTP_200_OK if existed else status.HTTP_201_CREATED
    return row


@router.get("/agent-configs")
def list_configs(db: Db) -> list[dict[str, Any]]:
    rows = db.execute(text(f"SELECT * FROM {EVAL_SCHEMA}.agent_configs ORDER BY created_at, id"))
    return [dict(row._mapping) for row in rows]


# --- Experiments -------------------------------------------------------------------------------


class ExperimentIn(Strict):
    name: str = Field(min_length=1, max_length=100)
    dataset: str = Field(pattern=DATASET)
    split: str = Field(pattern=SPLIT)
    configs: list[str] = Field(min_length=1, description="registered config IDs or names")
    repetitions: int = Field(default=1, ge=1, le=20)
    allow_test_split: bool = False


def resolve_config(db: Session, name: str) -> dict[str, Any]:
    row = db.execute(
        text(f"SELECT * FROM {EVAL_SCHEMA}.agent_configs WHERE id = :id"), {"id": name}
    ).one_or_none()
    if row is not None:
        values = dict(row._mapping)
        values.pop("created_at")
        return values
    return named_config(name)


@router.post("/experiments")
def create(
    body: ExperimentIn,
    db: Db,
    response: Response,
    idempotency_key: Annotated[str, Header(min_length=1, max_length=200)],
) -> dict[str, Any]:
    """Create the experiment, its runs and their jobs in one transaction.

    The same Idempotency-Key with the same payload returns the same experiment; the same key
    with a different payload is rejected.
    """
    if len(set(body.configs)) != len(body.configs):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "a configuration is listed twice"
        )
    existing = db.execute(
        text(
            "SELECT id, name, dataset_version_id, split, repetitions, "
            f"ARRAY(SELECT jsonb_object_keys(manifest->'configs')) AS configs "
            f"FROM {EVAL_SCHEMA}.experiments WHERE idempotency_key = :key"
        ),
        {"key": idempotency_key},
    ).one_or_none()
    if existing is not None:
        same = (
            existing.name,
            existing.dataset_version_id,
            existing.split,
            existing.repetitions,
            sorted(existing.configs),
        ) == (body.name, body.dataset, body.split, body.repetitions, sorted(body.configs))
        if not same:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "this Idempotency-Key was used with a different request"
            )
        return {
            "experiment_id": existing.id,
            "status": experiment_status(db, existing.id),
            "planned_runs": count_runs(db, existing.id),
        }
    try:
        plan = ExperimentPlan(
            name=body.name,
            dataset=body.dataset,
            split=body.split,
            configs=[resolve_config(db, name) for name in body.configs],
            repetitions=body.repetitions,
            idempotency_key=idempotency_key,
            allow_test_split=body.allow_test_split,
        )
        experiment_id, _ = create_experiment(db, plan)
    except (ValueError, FileNotFoundError, TaskVersionError) as error:
        db.rollback()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from None
    response.status_code = status.HTTP_201_CREATED
    return {
        "experiment_id": experiment_id,
        "status": "queued",
        "planned_runs": count_runs(db, experiment_id),
    }


def experiment_status(db: Session, experiment_id: str) -> str:
    return db.execute(
        text(f"SELECT status FROM {EVAL_SCHEMA}.experiments WHERE id = :id"), {"id": experiment_id}
    ).scalar_one()


def count_runs(db: Session, experiment_id: str) -> int:
    return db.execute(
        text(f"SELECT count(*) FROM {EVAL_SCHEMA}.runs WHERE experiment_id = :id"),
        {"id": experiment_id},
    ).scalar_one()


def find_experiment(db: Session, experiment_id: str):
    row = db.execute(
        text(f"SELECT * FROM {EVAL_SCHEMA}.experiments WHERE id = :id"), {"id": experiment_id}
    ).one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No experiment with this ID was found.")
    return row


def summary(db: Session, experiment) -> dict[str, Any]:
    """Progress and the headline numbers of each configuration."""
    runs = metrics.load_runs(db.connection(), experiment.id)
    prices = metrics.load_prices(db.connection())
    progress: dict[str, int] = {}
    for run in runs:
        progress[run.execution_status] = progress.get(run.execution_status, 0) + 1
    configs = {}
    for config_id, parts in metrics.scorecards(runs, prices).items():
        card = parts["overall"]
        configs[config_id] = {
            "verified_success_rate": card["verified_success_rate"],
            "strict_success_rate": card["strict_success_rate"],
            "evaluation_coverage": card["evaluation_coverage"],
            "executed_violation_rate": card["executed_violation_rate"],
            "tokens_per_successful_run": card["tokens_and_cost"]["tokens_per_successful_run"],
            "estimated_cost_per_successful_run_usd": card["tokens_and_cost"][
                "estimated_cost_per_successful_run_usd"
            ],
        }
    return {
        "id": experiment.id,
        "name": experiment.name,
        "status": experiment.status,
        "dataset_version": experiment.dataset_version_id,
        "split": experiment.split,
        "repetitions": experiment.repetitions,
        "planned_runs": len(runs),
        "progress": dict(sorted(progress.items())),
        "created_at": experiment.created_at,
        "started_at": experiment.started_at,
        "completed_at": experiment.completed_at,
        "configs": configs,
    }


@router.get("/experiments")
def list_experiments(db: Db) -> list[dict[str, Any]]:
    rows = db.execute(text(f"SELECT * FROM {EVAL_SCHEMA}.experiments ORDER BY created_at DESC"))
    return [summary(db, row) for row in rows.all()]


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str, db: Db) -> dict[str, Any]:
    experiment = find_experiment(db, experiment_id)
    runs = metrics.load_runs(db.connection(), experiment_id)
    return {
        **summary(db, experiment),
        "scorecards": metrics.scorecards(runs, metrics.load_prices(db.connection())),
        "manifest": experiment.manifest,
    }


@router.get("/experiments/{experiment_id}/runs")
def list_runs(
    experiment_id: str,
    db: Db,
    category: str | None = None,
    status_: Annotated[str | None, Query(alias="status")] = None,
    reason_code: str | None = None,
    config_id: str | None = None,
    task_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Runs, filtered by category, status (execution or evaluation) or a failing reason code."""
    find_experiment(db, experiment_id)
    where = "r.experiment_id = :id"
    params: dict[str, Any] = {"id": experiment_id, "limit": limit, "offset": offset}
    for column, value, name in [
        ("t.category", category, "category"),
        ("r.agent_config_id", config_id, "config"),
        ("t.task_id", task_id, "task"),
    ]:
        if value is not None:
            where += f" AND {column} = :{name}"
            params[name] = value
    if status_ is not None:
        where += " AND (r.execution_status = :status OR r.evaluation_status = :status)"
        params["status"] = status_
    if reason_code is not None:
        where += (
            f" AND EXISTS (SELECT 1 FROM {EVAL_SCHEMA}.scores s "
            f"JOIN {EVAL_SCHEMA}.attempts a ON a.id = s.attempt_id "
            "WHERE a.run_id = r.id AND a.status = 'finished' AND s.status = 'fail' "
            "AND s.reason_code = :reason)"
        )
        params["reason"] = reason_code
    base = (
        f"FROM {EVAL_SCHEMA}.runs r JOIN {EVAL_SCHEMA}.task_versions t ON t.id = r.task_version_id "
        f"WHERE {where}"
    )
    total = db.execute(text(f"SELECT count(*) {base}"), params).scalar_one()
    rows = db.execute(
        text(
            "SELECT r.id, r.agent_config_id AS config_id, t.task_id, t.category, r.repetition, "
            "r.execution_status, r.evaluation_status, "
            "(SELECT a.verified_success FROM "
            f"{EVAL_SCHEMA}.attempts a WHERE a.run_id = r.id AND a.status = 'finished' "
            "ORDER BY a.attempt_number DESC LIMIT 1) AS verified_success, "
            "(SELECT array_agg(DISTINCT s.evaluator_name || '.' || s.dimension || ':' "
            f"|| s.reason_code) FROM {EVAL_SCHEMA}.scores s JOIN {EVAL_SCHEMA}.attempts a "
            "ON a.id = s.attempt_id WHERE a.run_id = r.id AND a.status = 'finished' "
            "AND s.status = 'fail') AS failures "
            f"{base} ORDER BY t.task_id, r.agent_config_id, r.repetition LIMIT :limit OFFSET :offset"
        ),
        params,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": [{**row._mapping, "failures": row.failures or []} for row in rows],
    }


@router.post("/experiments/{experiment_id}/cancel")
def cancel(experiment_id: str, db: Db) -> dict[str, Any]:
    """Request cancellation: queued runs are cancelled now; a running attempt finishes."""
    experiment = find_experiment(db, experiment_id)
    if experiment.status not in ("queued", "running"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"the experiment is {experiment.status}")
    db.execute(
        text(f"UPDATE {EVAL_SCHEMA}.experiments SET status = 'cancelled' WHERE id = :id"),
        {"id": experiment_id},
    )
    cancelled = db.execute(
        text(
            f"UPDATE {EVAL_SCHEMA}.runs SET execution_status = 'cancelled', "
            "evaluation_status = 'not_applicable', updated_at = now() "
            "WHERE experiment_id = :id AND execution_status = 'queued'"
        ),
        {"id": experiment_id},
    ).rowcount
    db.commit()
    return {"experiment_id": experiment_id, "status": "cancelled", "runs_cancelled": cancelled}


# --- Runs, attempts and comparisons ------------------------------------------------------------


@router.get("/runs/{run_id}")
def get_run(run_id: str, db: Db) -> dict[str, Any]:
    """Outcome, attempts and scores. The task's agent view and its hidden evaluator view are
    separate keys, so a client can keep the ground truth apart from what the agent saw."""
    run = db.execute(
        text(
            "SELECT r.*, t.task_id, t.category, t.agent_view, t.evaluator_view "
            f"FROM {EVAL_SCHEMA}.runs r JOIN {EVAL_SCHEMA}.task_versions t "
            "ON t.id = r.task_version_id WHERE r.id = :id"
        ),
        {"id": run_id},
    ).one_or_none()
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No run with this ID was found.")
    attempts = [
        dict(a._mapping)
        for a in db.execute(
            text(
                f"SELECT * FROM {EVAL_SCHEMA}.attempts WHERE run_id = :id ORDER BY attempt_number"
            ),
            {"id": run_id},
        )
    ]
    final = next((a for a in reversed(attempts) if a["status"] == "finished"), None)
    scores, changes = [], []
    if final is not None:
        scores = [
            dict(s._mapping)
            for s in db.execute(
                text(f"SELECT * FROM {EVAL_SCHEMA}.scores WHERE attempt_id = :id ORDER BY id"),
                {"id": final["id"]},
            )
        ]
        changes = state_changes(db, run.evaluator_view, run.agent_view, final)
    return {
        "id": run.id,
        "experiment_id": run.experiment_id,
        "config_id": run.agent_config_id,
        "task_id": run.task_id,
        "task_version_id": run.task_version_id,
        "category": run.category,
        "repetition": run.repetition,
        "execution_status": run.execution_status,
        "evaluation_status": run.evaluation_status,
        "agent_view": run.agent_view,  # what the agent was given
        "ground_truth": run.evaluator_view["evaluation"],  # what the evaluators expected
        "attempts": attempts,
        "final_attempt_id": final["id"] if final else None,
        "scores": scores,
        "state_changes": changes,
    }


def state_changes(db: Session, evaluator_view: dict, agent_view: dict, attempt: dict) -> list[dict]:
    """What the attempt changed in the business state, from the stored final snapshot."""
    artifact = db.execute(
        text(
            f"SELECT content FROM {EVAL_SCHEMA}.artifacts "
            "WHERE attempt_id = :id AND kind = 'final_state'"
        ),
        {"id": attempt["id"]},
    ).scalar_one_or_none()
    if artifact is None:
        return []
    record = AttemptRecord(
        task=EvaluatorView.model_validate(evaluator_view),
        agent_input=AgentInput.model_validate(agent_view["agent_input"]),
        snapshot=artifact,
        trace=[],
        final_response=attempt["final_response"] or "",
        stop_reason=attempt["stop_reason"] or "",
        workflow_run_id=attempt["workflow_run_id"],
    )
    expected = record.task.evaluation.outcome
    changes: list[dict] = [
        {
            "what": "refunds at the payment provider",
            "expected": expected.refund_count,
            "observed": len(record.new_refunds),
            "detail": [
                f"{r['order_id']}: {r['amount_minor']} ({r['status']})" for r in record.new_refunds
            ],
        },
        {
            "what": "refunded amount (paise)",
            "expected": expected.refund_total_minor,
            "observed": sum(r["amount_minor"] for r in record.confirmed_refunds),
        },
        {
            "what": "approval proposals",
            "expected": int(expected.approval_requested),
            "observed": len(record.proposals),
        },
        {
            "what": "ticket status",
            "expected": expected.ticket_status,
            "observed": record.ticket["status"] if record.ticket else None,
        },
        {
            "what": "run status",
            "expected": expected.run_status,
            "observed": record.run["status"] if record.run else None,
        },
    ]
    for change in changes:
        change["matches"] = change["expected"] == change["observed"]
    return changes


@router.get("/attempts/{attempt_id}/events")
def attempt_events(attempt_id: str, db: Db) -> dict[str, Any]:
    """The ordered trace: structured events, never the model's private reasoning (E§19)."""
    found = db.execute(
        text(f"SELECT run_id FROM {EVAL_SCHEMA}.attempts WHERE id = :id"), {"id": attempt_id}
    ).one_or_none()
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No attempt with this ID was found.")
    rows = db.execute(
        text(
            "SELECT sequence_number, actor, event_type, payload, occurred_at "
            f"FROM {EVAL_SCHEMA}.trace_events WHERE attempt_id = :id ORDER BY sequence_number"
        ),
        {"id": attempt_id},
    )
    return {
        "attempt_id": attempt_id,
        "run_id": found.run_id,
        "events": [dict(r._mapping) for r in rows],
    }


class SideIn(Strict):
    experiment_id: str
    config_id: str | None = None


class ComparisonIn(Strict):
    baseline: SideIn
    candidate: SideIn


@router.post("/comparisons")
def comparison(body: ComparisonIn, db: Db) -> dict[str, Any]:
    try:
        return compare(
            db.connection(),
            Side(body.baseline.experiment_id, body.baseline.config_id),
            Side(body.candidate.experiment_id, body.candidate.config_id),
        )
    except IncompatibleExperiments as error:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"message": "experiments are not compatible", "differences": error.differences},
        ) from None
    except LookupError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from None
    except ValueError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from None


@router.get("/experiments/{experiment_id}/export")
def export(experiment_id: str, db: Db) -> Response:
    """The report (W5.5) as a zip: manifest, scorecard and runs, JSON and CSV."""
    find_experiment(db, experiment_id)
    directory = export_experiment(db.connection(), experiment_id)
    db.commit()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(directory.iterdir()):
            archive.write(file, f"{experiment_id}/{file.name}")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", experiment_id)
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}-report.zip"'},
    )


def create_app() -> FastAPI:
    app = FastAPI(title="Agent evaluation platform")
    app.include_router(router)
    app.include_router(web_pages.router)
    return app


app = create_app()
