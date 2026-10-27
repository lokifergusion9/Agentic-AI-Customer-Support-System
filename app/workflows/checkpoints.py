"""LangGraph checkpoints in PostgreSQL, in their own `langgraph` schema (W1.4, W2.13, W3.3).

Checkpoints let a paused run (waiting for the customer, later for a reviewer) resume on the
same thread. They are not the financial ledger: refund state lives in the `support` tables.
"""

from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.engine import make_url

from app.config import get_settings

CHECKPOINT_SCHEMA = "langgraph"


@lru_cache
def get_checkpointer() -> BaseCheckpointSaver:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg import Connection
    from psycopg.rows import dict_row

    url = make_url(get_settings().database_url).set(drivername="postgresql")
    connection = Connection.connect(
        url.render_as_string(hide_password=False),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
        connect_timeout=5,
    )
    connection.execute(f"CREATE SCHEMA IF NOT EXISTS {CHECKPOINT_SCHEMA}")
    connection.execute(f"SET search_path TO {CHECKPOINT_SCHEMA}")
    saver = PostgresSaver(connection)  # type: ignore[arg-type]
    saver.setup()
    return saver
