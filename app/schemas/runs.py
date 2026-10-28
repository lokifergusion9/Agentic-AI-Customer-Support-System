"""Request and response bodies for the run endpoints (W2.14)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.schemas.tickets import CustomerMessage


class RunMessage(BaseModel):
    """POST /runs/{run_id}/messages: the customer's answer to a clarification question."""

    model_config = ConfigDict(extra="forbid")

    message: CustomerMessage


class RunEventView(BaseModel):
    """One redacted audit event: IDs, statuses, amounts and counts, never message text."""

    id: int
    # 1, 2, 3, ... within the run: the exact replay order (W3.15).
    sequence_number: int
    created_at: datetime
    actor: str
    event_type: str
    payload: dict[str, Any]
