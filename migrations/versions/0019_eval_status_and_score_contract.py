"""Eval statuses and the score contract as W4.8 and W4.9 define them.

- runs: execution status (queued, running, completed, timed_out, budget_exceeded,
  infrastructure_error, cancelled) apart from evaluation status (pending, pass, fail, partial,
  not_applicable, evaluation_error). A confident wrong answer is completed + fail.
- attempts: `interrupted` for an attempt a lost worker left behind (it is replaced, W4.8),
  the attempt's own execution status, and verified success (E§20; null when unknown).
- scores: the score contract fields (applicable, status pass | fail | partial |
  not_applicable | evaluation_error, reason_code, evidence) replace passed / reason / details.

The eval tables hold no data yet, so the old values need no conversion.

Revision ID: 0019
Revises: 0018
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from evals.db import EVAL_SCHEMA as S

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EXECUTION = (
    "'queued', 'running', 'completed', 'timed_out', 'budget_exceeded', "
    "'infrastructure_error', 'cancelled'"
)
SCORE_STATUS = "'pass', 'fail', 'partial', 'not_applicable', 'evaluation_error'"
EVALUATION = f"'pending', {SCORE_STATUS}"


def replace_check(table: str, name: str, condition: str) -> None:
    op.drop_constraint(name, table, schema=S, type_="check")
    op.create_check_constraint(name, table, condition, schema=S)


def upgrade() -> None:
    replace_check("runs", "ck_runs_execution_status", f"execution_status IN ({EXECUTION})")
    replace_check("runs", "ck_runs_evaluation_status", f"evaluation_status IN ({EVALUATION})")

    op.add_column("attempts", sa.Column("execution_status", sa.Text(), nullable=True), schema=S)
    op.add_column("attempts", sa.Column("verified_success", sa.Boolean(), nullable=True), schema=S)
    replace_check(
        "attempts",
        "ck_attempts_status",
        "status IN ('running', 'finished', 'interrupted', 'error')",
    )
    op.create_check_constraint(
        "ck_attempts_execution_status",
        "attempts",
        f"execution_status IS NULL OR execution_status IN ({EXECUTION})",
        schema=S,
    )

    op.drop_constraint("ck_scores_passed", "scores", schema=S, type_="check")
    op.drop_column("scores", "passed", schema=S)
    op.alter_column("scores", "reason", new_column_name="reason_code", schema=S)
    op.alter_column("scores", "details", new_column_name="evidence", schema=S)
    op.add_column(
        "scores",
        sa.Column("applicable", sa.Boolean(), nullable=False, server_default=sa.true()),
        schema=S,
    )
    replace_check("scores", "ck_scores_status", f"status IN ({SCORE_STATUS})")
    op.create_check_constraint(
        "ck_scores_applicable",
        "scores",
        "applicable = (status <> 'not_applicable')",
        schema=S,
    )


def downgrade() -> None:
    # Results recorded under the new statuses cannot be expressed in the old ones.
    for table in ("scores", "artifacts", "trace_events", "attempts", "runs"):
        op.execute(f"DELETE FROM {S}.{table}")
    op.drop_constraint("ck_scores_applicable", "scores", schema=S, type_="check")
    replace_check("scores", "ck_scores_status", "status IN ('scored', 'not_applicable', 'error')")
    op.drop_column("scores", "applicable", schema=S)
    op.alter_column("scores", "evidence", new_column_name="details", schema=S)
    op.alter_column("scores", "reason_code", new_column_name="reason", schema=S)
    op.add_column("scores", sa.Column("passed", sa.Boolean(), nullable=True), schema=S)
    op.create_check_constraint(
        "ck_scores_passed", "scores", "(status = 'scored') = (passed IS NOT NULL)", schema=S
    )

    op.drop_constraint("ck_attempts_execution_status", "attempts", schema=S, type_="check")
    replace_check("attempts", "ck_attempts_status", "status IN ('running', 'finished', 'error')")
    op.drop_column("attempts", "verified_success", schema=S)
    op.drop_column("attempts", "execution_status", schema=S)

    replace_check(
        "runs",
        "ck_runs_execution_status",
        "execution_status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
    )
    replace_check(
        "runs",
        "ck_runs_evaluation_status",
        "evaluation_status IN ('pending', 'scored', 'failed', 'skipped')",
    )
