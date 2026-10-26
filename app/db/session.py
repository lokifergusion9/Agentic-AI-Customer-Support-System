"""SQLAlchemy engine and the per-request session dependency."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    # A short connect timeout so /health reports a down database instead of hanging.
    return create_engine(
        get_settings().database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5}
    )


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, closed afterwards."""
    with Session(get_engine()) as session:
        yield session
