"""Reproducible report export (W5.5, E§1, E§22).

    uv run python -m evals.reports export <experiment_id>

Writes `evals/reports/<experiment_id>/`:

- `manifest.json`: everything needed to rerun and to check what ran: dataset version and hash,
  the configurations (adapter, model, prompt hashes, limits, tool-schema hash), the model IDs
  the provider reported it served, code commit, lockfile hash, fault configuration, and the
  environment and evaluator versions.
- `scorecard.json` and `scorecard.csv`: the metrics of W5.1 per configuration and category.
- `runs.json` and `runs.csv`: one row per run.

The files hold no export time, so exporting an unchanged experiment again gives the same bytes.
Each file is also recorded in `eval.artifacts` with its hash.
"""

import argparse
import csv
import hashlib
import io
import json
import secrets
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.db.session import get_engine
from evals import metrics
from evals.db import EVAL_SCHEMA
from evals.metrics import RunData

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

SCORECARD_COLUMNS = [
    ("verified_success_rate", ["verified_success_rate", "value"]),
    ("verified_success_n", ["verified_success_rate", "n"]),
    ("scorable_runs", ["scorable_runs"]),
    ("strict_success_rate", ["strict_success_rate", "value"]),
    ("planned_runs", ["planned_runs"]),
    ("evaluation_coverage", ["evaluation_coverage", "value"]),
    ("unsafe_attempt_rate", ["unsafe_attempt_rate", "value"]),
    ("executed_violation_rate", ["executed_violation_rate", "value"]),
    ("duplicate_refund_count", ["duplicate_refund_count"]),
    ("tool_schema_validity", ["tool_schema_validity", "value"]),
    ("tool_call_precision", ["tool_call_precision", "value"]),
    ("required_call_recall", ["required_call_recall", "value"]),
    ("tool_calls_per_run", ["tool_calls_per_run", "value"]),
    ("policy_retrieval_any", ["policy_retrieval", "any_required_section", "value"]),
    ("policy_retrieval_all", ["policy_retrieval", "all_required_sections", "value"]),
    ("active_ms_p50", ["latency", "active_ms", "p50"]),
    ("active_ms_p95", ["latency", "active_ms", "p95"]),
    ("approval_wait_ms_p50", ["latency", "approval_wait_ms", "p50"]),
    ("queue_ms_p50", ["latency", "queue_ms", "p50"]),
    ("model_calls_per_run", ["tokens_and_cost", "model_calls_per_run", "value"]),
    ("tokens_per_successful_run", ["tokens_and_cost", "tokens_per_successful_run"]),
    ("runs_with_unknown_usage", ["tokens_and_cost", "runs_with_unknown_usage"]),
    ("estimated_cost_usd", ["tokens_and_cost", "estimated_cost_usd"]),
    ("runs_with_unknown_cost", ["tokens_and_cost", "runs_with_unknown_cost"]),
    ("mean_task_success", ["repetitions", "mean_task_success"]),
]


def dig(value: Any, path: list[str]) -> Any:
    for key in path:
        value = value[key]
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=False) + "\n"


def build_manifest(
    connection: Connection, experiment_id: str, runs: list[RunData]
) -> dict[str, Any]:
    experiment = connection.execute(
        text(
            "SELECT e.id, e.name, e.idempotency_key, e.dataset_version_id, e.split, "
            "e.repetitions, e.manifest, e.status, d.content_hash AS dataset_hash "
            f"FROM {EVAL_SCHEMA}.experiments e "
            f"JOIN {EVAL_SCHEMA}.dataset_versions d ON d.id = e.dataset_version_id "
            "WHERE e.id = :id"
        ),
        {"id": experiment_id},
    ).one_or_none()
    if experiment is None:
        raise LookupError(f"no experiment {experiment_id!r}")
    config_ids = sorted({r.config_id for r in runs})
    configs = {
        row.id: {
            key: value for key, value in row._mapping.items() if key not in ("id", "created_at")
        }
        for row in connection.execute(
            text(f"SELECT * FROM {EVAL_SCHEMA}.agent_configs WHERE id = ANY(:ids)"),
            {"ids": config_ids},
        )
    }
    resolved = connection.execute(
        text(
            "SELECT r.agent_config_id AS config_id, "
            "array_agg(DISTINCT e.payload->>'resolved_model') AS models "
            f"FROM {EVAL_SCHEMA}.trace_events e JOIN {EVAL_SCHEMA}.attempts a ON a.id = e.attempt_id "
            f"JOIN {EVAL_SCHEMA}.runs r ON r.id = a.run_id "
            "WHERE r.experiment_id = :id AND e.event_type = 'model_call' "
            "AND e.payload->>'outcome' = 'ok' GROUP BY 1"
        ),
        {"id": experiment_id},
    )
    for row in resolved:
        configs[row.config_id]["resolved_model_ids"] = sorted(m for m in row.models if m)
    faults = {
        task.task_id: task.task["environment_fixture"]["faults"]
        for task in {r.task_id: r for r in runs}.values()
        if task.task["environment_fixture"]["faults"]
    }
    return {
        "experiment": {
            "id": experiment.id,
            "name": experiment.name,
            "idempotency_key": experiment.idempotency_key,
            "status": experiment.status,
            "dataset_version": experiment.dataset_version_id,
            "dataset_hash": experiment.dataset_hash,
            "split": experiment.split,
            "repetitions": experiment.repetitions,
        },
        "configs": configs,
        "fault_configuration": faults,
        "manifest": experiment.manifest,
    }


def run_row(run: RunData, prices: list[dict[str, Any]]) -> dict[str, Any]:
    final = run.final or {}
    usage = final.get("usage") or {}
    tokens = metrics.run_tokens(run)
    calls = metrics.agent_calls(run)
    queue_ms = (
        None
        if run.first_started_at is None
        else round((run.first_started_at - run.created_at).total_seconds() * 1000)
    )
    return {
        "run_id": run.run_id,
        "config_id": run.config_id,
        "task_id": run.task_id,
        "task_version": run.task_version_id,
        "category": run.category,
        "repetition": run.repetition,
        "execution_status": run.execution_status,
        "evaluation_status": run.evaluation_status,
        "verified_success": final.get("verified_success"),
        "human_assisted": run.human_assisted,
        "stop_reason": final.get("stop_reason"),
        "attempt_id": final.get("id"),
        "model_calls": usage.get("model_calls"),
        "input_tokens": tokens["input"],
        "output_tokens": tokens["output"],
        "tool_calls": len(calls),
        "queue_ms": queue_ms,
        "active_ms": final.get("active_ms"),
        "approval_wait_ms": final.get("approval_wait_ms"),
        "estimated_cost_usd": metrics.run_cost(run, prices),
        "failures": "; ".join(
            sorted(
                f"{s['evaluator_name']}.{s['dimension']}:{s['reason_code']}"
                for s in run.scores
                if s["status"] == "fail"
            )
        ),
    }


def to_csv(rows: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    if rows:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return out.getvalue()


def scorecard_rows(cards: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for config_id, parts in cards.items():
        for category, card in [("ALL", parts["overall"]), *sorted(parts["by_category"].items())]:
            row: dict[str, Any] = {"config_id": config_id, "category": category}
            for name, path in SCORECARD_COLUMNS:
                row[name] = dig(card, path)
            rows.append(row)
    return rows


def export_experiment(
    connection: Connection, experiment_id: str, out_dir: Path | None = None
) -> Path:
    """Write the report files and record them as artifacts; returns the directory."""
    runs = metrics.load_runs(connection, experiment_id)
    prices = metrics.load_prices(connection)
    cards = metrics.scorecards(runs, prices)
    rows = [run_row(r, prices) for r in runs]
    files = {
        "manifest.json": canonical(build_manifest(connection, experiment_id, runs)),
        "scorecard.json": canonical(cards),
        "scorecard.csv": to_csv(scorecard_rows(cards)),
        "runs.json": canonical(rows),
        "runs.csv": to_csv(rows),
    }
    target = (out_dir or REPORTS_DIR) / experiment_id
    target.mkdir(parents=True, exist_ok=True)
    connection.execute(
        text(
            f"DELETE FROM {EVAL_SCHEMA}.artifacts WHERE experiment_id = :id "
            "AND kind IN ('report_json', 'report_csv')"
        ),
        {"id": experiment_id},
    )
    for name, content in files.items():
        (target / name).write_text(content, encoding="utf-8", newline="\n")
        connection.execute(
            text(
                f"INSERT INTO {EVAL_SCHEMA}.artifacts (id, experiment_id, kind, path, content_hash) "
                "VALUES (:id, :experiment, :kind, :path, :hash)"
            ),
            {
                "id": f"art_{secrets.token_hex(8)}",
                "experiment": experiment_id,
                "kind": "report_json" if name.endswith(".json") else "report_csv",
                "path": str(target / name),
                "hash": "sha256:" + hashlib.sha256(content.encode()).hexdigest(),
            },
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("experiment_id")
    export.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    with Session(get_engine()) as db:
        target = export_experiment(db.connection(), args.experiment_id, args.out)
        db.commit()
    print(f"report written to {target}")


if __name__ == "__main__":
    main()
