"""Append-only audit trail (W1.15 basic; every transition and tool call in W3.15).

Payloads are redacted by the caller: IDs, amounts and statuses only, never message text,
credentials or tokens.
"""

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SCHEMA


def record_event(
    db: Session,
    *,
    actor: str,
    event_type: str,
    payload: dict[str, Any],
    ticket_id: str | None = None,
    run_id: str | None = None,
) -> None:
    """Add an audit event in the caller's transaction. `actor` is kind:id, e.g. customer:cus_1."""
    db.execute(
        text(
            f"INSERT INTO {SCHEMA}.audit_events "
            "(ticket_id, run_id, actor, event_type, redacted_payload) "
            "VALUES (:ticket, :run, :actor, :event_type, CAST(:payload AS jsonb))"
        ),
        {
            "ticket": ticket_id,
            "run": run_id,
            "actor": actor,
            "event_type": event_type,
            "payload": json.dumps(payload),
        },
    )
