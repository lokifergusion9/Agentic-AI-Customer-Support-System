"""Request and response bodies for POST /tickets (W1.7), and the ticket read result (W1.12)."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

CustomerMessage = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]
OrderId = Annotated[str, StringConstraints(pattern=r"^ord_[A-Za-z0-9]{1,32}$")]


class TicketCreate(BaseModel):
    # No customer_id field: the customer comes from the token. Unknown fields are
    # rejected so a client cannot try to pass one.
    model_config = ConfigDict(extra="forbid")

    message: CustomerMessage
    # Optional: the agent asks for it later if it is missing (W2.13).
    order_id: OrderId | None = None


class TicketResponse(BaseModel):
    id: str
    order_id: str | None
    status: str
    category: str | None
    created_at: datetime


class TicketView(BaseModel):
    """A ticket as the local record has it. The helpdesk copy is synced in W3.14."""

    id: str
    order_id: str | None
    category: str | None
    status: str
    customer_message: str
    summary: str | None
    external_ticket_ref: str | None
    created_at: datetime
    updated_at: datetime
