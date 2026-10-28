"""The pieces every tool shares: server-side context, results and tool specs (W2.4, W2.5).

Identity never comes from the model. The workflow builds a ToolContext from the authenticated
request (the customer, the run, the ticket) and hands it to every tool; the model only supplies
the narrow arguments in each tool's Pydantic model, which forbids anything extra.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.policies.embeddings import Embedder
from app.services.errors import ServiceFailure
from app.services.providers import ProviderClient


@dataclass
class ToolContext:
    db: Session
    customer: AuthenticatedCustomer
    run_id: str
    ticket_id: str
    payments: ProviderClient
    shipping: ProviderClient
    embedder: Embedder
    # Store date for eligibility decisions; None means today (tests and evals pin it).
    today: date | None = None
    # The helpdesk (W3.14). None keeps ticket updates local only (unit and agent tests).
    ticketing: ProviderClient | None = None
    # Shared with the run's RunRecorder: agents running in parallel threads (W2.10) take
    # turns on the one database session.
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def actor(self) -> str:
        return f"customer:{self.customer.customer_id}"


class ToolArgs(BaseModel):
    """Base for tool arguments. Extra fields are rejected, so the model cannot add a customer
    ID, a reviewer, an approval status or credentials. Lax mode stays on for numbers because
    model providers send JSON numbers as floats (249900.0 is accepted as 249900)."""

    model_config = ConfigDict(extra="forbid")


# read: facts only. propose: stores a proposal, never moves money. execute: code-owned steps
# the model is never offered (refund execution, ticket updates).
Access = Literal["read", "propose", "execute"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    handler: Callable[[ToolContext, Any], BaseModel | list[BaseModel] | dict[str, Any]]
    access: Access
    # Free-text arguments, audited as their length only (customer text stays out of the trail).
    free_text_args: frozenset[str] = frozenset()

    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


def ok(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data}


def failure(error: ServiceFailure) -> dict[str, Any]:
    """The W1.13 failure shape the model sees."""
    return {
        "success": False,
        "error_code": error.error_code,
        "retryable": error.retryable,
        "message": error.message,
    }
