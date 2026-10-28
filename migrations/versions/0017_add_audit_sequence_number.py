"""Add audit_events.sequence_number: a run's events replay in exact order (W3.15, E§19).

Numbered 1, 2, 3, ... per run, unique per run; events without a run (a ticket created before
any run) have none. A BEFORE INSERT trigger assigns the next number under a per-run
transaction lock, so two writers for the same run (the API and the worker, say) never take the
same number, whoever inserts. Existing rows are numbered in insertion order, with the
append-only trigger paused for it.

Revision ID: 0017
Revises: 0016
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db import SCHEMA

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TRIGGER = "trg_audit_events_append_only"


def upgrade() -> None:
    op.add_column(
        "audit_events", sa.Column("sequence_number", sa.Integer(), nullable=True), schema=SCHEMA
    )
    op.execute(f"ALTER TABLE {SCHEMA}.audit_events DISABLE TRIGGER {TRIGGER}")
    op.execute(
        f"""
        UPDATE {SCHEMA}.audit_events e SET sequence_number = numbered.n
        FROM (
            SELECT id, row_number() OVER (PARTITION BY run_id ORDER BY id) AS n
            FROM {SCHEMA}.audit_events WHERE run_id IS NOT NULL
        ) numbered
        WHERE e.id = numbered.id
        """
    )
    op.execute(f"ALTER TABLE {SCHEMA}.audit_events ENABLE TRIGGER {TRIGGER}")
    op.execute(
        f"""
        CREATE FUNCTION {SCHEMA}.number_audit_event() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.run_id IS NOT NULL AND NEW.sequence_number IS NULL THEN
                -- Held until commit: the next writer for this run waits, then sees our row.
                PERFORM pg_advisory_xact_lock(hashtextextended('audit:' || NEW.run_id, 0));
                SELECT COALESCE(max(sequence_number), 0) + 1 INTO NEW.sequence_number
                FROM {SCHEMA}.audit_events WHERE run_id = NEW.run_id;
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    op.execute(
        f"CREATE TRIGGER trg_audit_events_sequence BEFORE INSERT ON {SCHEMA}.audit_events "
        f"FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.number_audit_event()"
    )
    op.create_unique_constraint(
        "uq_audit_events_run_sequence",
        "audit_events",
        ["run_id", "sequence_number"],
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_audit_events_sequence_number",
        "audit_events",
        "(run_id IS NULL) = (sequence_number IS NULL) AND (sequence_number IS NULL "
        "OR sequence_number > 0)",
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint("ck_audit_events_sequence_number", "audit_events", schema=SCHEMA)
    op.drop_constraint("uq_audit_events_run_sequence", "audit_events", schema=SCHEMA)
    op.execute(f"DROP TRIGGER trg_audit_events_sequence ON {SCHEMA}.audit_events")
    op.execute(f"DROP FUNCTION {SCHEMA}.number_audit_event()")
    op.drop_column("audit_events", "sequence_number", schema=SCHEMA)
