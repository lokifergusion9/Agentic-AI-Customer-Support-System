"""Alembic environment. Migrations are hand-written and create tables in the app schema."""

from alembic import context
from sqlalchemy import create_engine, text

from app.config import get_settings
from app.db import SCHEMA


def database_url() -> str:
    # Tests point Alembic at a separate database through the config; otherwise use settings.
    return context.config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(url=database_url(), literal_binds=True, version_table_schema=SCHEMA)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(database_url())
    with engine.connect() as connection:
        # The version table lives in the app schema, so the schema must exist before migrating.
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
        connection.commit()
        context.configure(connection=connection, version_table_schema=SCHEMA)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
