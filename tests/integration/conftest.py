"""Fixtures for integration tests that need PostgreSQL (`docker compose up -d`)."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from app.config import get_settings

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    return config


@pytest.fixture(scope="session")
def database_url() -> str:
    """URL of a fresh, fully migrated test database, separate from the dev database."""
    url = make_url(
        os.environ.get("TEST_DATABASE_URL")
        or make_url(get_settings().database_url).set(database="support_test")
    )
    # The database is dropped and recreated, so never point this at a real one.
    assert url.database and url.database.endswith("_test"), "test database name must end in _test"
    admin = create_engine(
        url.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        # Fail fast, so the suite skips quickly when PostgreSQL is not running.
        connect_args={"connect_timeout": 3},
    )
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{url.database}" WITH (FORCE)'))
            connection.execute(text(f'CREATE DATABASE "{url.database}"'))
    except OperationalError as exc:
        pytest.skip(f"PostgreSQL is not reachable, start it with `docker compose up -d`: {exc}")
    finally:
        admin.dispose()

    rendered = url.render_as_string(hide_password=False)
    command.upgrade(alembic_config(rendered), "head")
    return rendered


@pytest.fixture(scope="session")
def alembic_cfg(database_url: str) -> Config:
    return alembic_config(database_url)


@pytest.fixture(scope="session")
def engine(database_url: str) -> Iterator[Engine]:
    engine = create_engine(database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    """A connection whose changes are rolled back after the test."""
    with engine.connect() as connection:
        transaction = connection.begin()
        yield connection
        transaction.rollback()
