"""Migrations rebuild the schema from scratch and tear it down cleanly (W1.3, W1.4)."""

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from app.db import SCHEMA

APP_TABLES = {
    "customers",
    "orders",
    "order_items",
    "returns",
    "tickets",
    "workflow_runs",
    "policy_documents",
    "policy_chunks",
    "action_proposals",
    "approvals",
    "refund_operations",
    "audit_events",
    "jobs",
}


def tables(engine: Engine, schema: str) -> set[str]:
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = :schema"),
            {"schema": schema},
        )
        return {row.table_name for row in rows}


def extensions(engine: Engine) -> set[str]:
    with engine.connect() as connection:
        return {row.extname for row in connection.execute(text("SELECT extname FROM pg_extension"))}


def test_downgrade_then_upgrade_reproduces_the_schema(alembic_cfg: Config, engine: Engine) -> None:
    command.downgrade(alembic_cfg, "base")
    assert tables(engine, SCHEMA) == {"alembic_version"}
    assert not {"vector", "btree_gist"} & extensions(engine)

    command.upgrade(alembic_cfg, "head")
    assert tables(engine, SCHEMA) == APP_TABLES | {"alembic_version"}
    assert {"vector", "btree_gist"} <= extensions(engine)


def test_app_tables_are_kept_out_of_the_public_schema(engine: Engine) -> None:
    # LangGraph checkpoint tables get their own schema (W3.3); nothing of ours lands in public.
    assert tables(engine, "public") == set()


def test_money_is_stored_in_integer_minor_units(engine: Engine) -> None:
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = :schema AND column_name LIKE '%amount%'"
            ),
            {"schema": SCHEMA},
        ).all()
    assert rows, "expected money columns"
    assert {(row.table_name, row.column_name, row.data_type) for row in rows} == {
        ("orders", "paid_amount_minor", "bigint"),
        ("orders", "shipping_amount_minor", "bigint"),
        ("order_items", "paid_amount_minor", "bigint"),
        ("refund_operations", "amount_minor", "bigint"),
    }
