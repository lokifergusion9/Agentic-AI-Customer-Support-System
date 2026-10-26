"""Typed result of the return read service (W1.12)."""

from datetime import datetime

from pydantic import BaseModel


class ReturnView(BaseModel):
    return_id: str
    order_id: str
    # requested | in_transit | received | closed
    status: str
    # The refund window is measured from delivery to this date (refund_policy section 3).
    requested_at: datetime
    received_at: datetime | None
    # pending | passed | failed, once the item is received (return_processing section 4)
    inspection_status: str | None
