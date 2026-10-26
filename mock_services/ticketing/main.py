"""Mock ticketing (helpdesk) API (W1.10): `uvicorn mock_services.ticketing.main:app --port 8003`.

The app's `tickets` table is the local record; its status changes only after this API
confirms (Q1). Every endpoint except `/health` needs the `X-API-Key` header (see
`mock_services.common`).

Both writes are safe to retry, so after a timeout the app can retry only the ticket update
(W3.14): creating a ticket again for the same `local_ticket_id`, or sending an update again
with the same `update_id` and body, returns the earlier result instead of applying it twice.
A changed body for a used ID is rejected.
"""

import secrets
import time
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, Response, status
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from mock_services.common import (
    Fault,
    FaultRequestBase,
    MockError,
    add_error_handler,
    add_fault_routes,
    api_key_check,
    raise_fault_error,
)
from mock_services.ticketing.store import STATUSES, Note, Ticket, Update, store


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local-only default matching .env.example.
    mock_ticketing_api_key: str = "mock-ticketing-local-key"


settings = Settings()

app = FastAPI(title="Mock ticketing provider")
add_error_handler(app)
api = APIRouter(dependencies=[Depends(api_key_check(settings.mock_ticketing_api_key))])

Status = Literal["open", "awaiting_customer", "awaiting_approval", "escalated", "resolved"]
assert set(Status.__args__) == set(STATUSES)


class NoteModel(BaseModel):
    text: str
    at: datetime


class TicketOut(BaseModel):
    ref: str
    local_ticket_id: str
    customer_id: str
    subject: str
    status: str
    created_at: datetime
    updated_at: datetime
    notes: list[NoteModel]


class UpdateOut(BaseModel):
    update_id: str
    ticket_ref: str
    status: str | None
    note: str | None
    applied_at: datetime


def ticket_out(ticket: Ticket) -> TicketOut:
    return TicketOut(
        ref=ticket.ref,
        local_ticket_id=ticket.local_ticket_id,
        customer_id=ticket.customer_id,
        subject=ticket.subject,
        status=ticket.status,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at,
        notes=[NoteModel(**n.__dict__) for n in ticket.notes],
    )


def take_write_fault() -> Fault | None:
    """Pop the next fault for a new write; raise it now if it is an error."""
    fault = store.faults.pop()
    if fault is not None and fault.mode == "error":
        raise_fault_error(fault)
    return fault


def hold_response(fault: Fault | None) -> None:
    # delay: the write is not applied until after the wait. accept_then_timeout: it was
    # applied before the wait, so a client that gives up never learns it succeeded.
    # Either way the wait happens outside the lock.
    if fault is not None and fault.mode in ("delay", "accept_then_timeout"):
        time.sleep(fault.delay_seconds)


def not_found() -> MockError:
    return MockError(status.HTTP_404_NOT_FOUND, "TICKET_NOT_FOUND", "No such ticket.")


# --- Admin ------------------------------------------------------------------------------


class SeedTicket(BaseModel):
    ref: str
    local_ticket_id: str
    customer_id: str
    subject: str
    status: Status
    created_at: datetime
    notes: list[NoteModel] = []


class ResetRequest(BaseModel):
    tickets: list[SeedTicket]


@api.post("/admin/reset")
def reset(request: ResetRequest) -> dict[str, int]:
    """Replace all state with the seed data and clear queued faults."""
    tickets = [
        Ticket(
            ref=t.ref,
            local_ticket_id=t.local_ticket_id,
            customer_id=t.customer_id,
            subject=t.subject,
            status=t.status,
            created_at=t.created_at,
            updated_at=max([t.created_at, *(n.at for n in t.notes)]),
            notes=[Note(**n.model_dump()) for n in t.notes],
        )
        for t in request.tickets
    ]
    store.reset(tickets)
    return {"tickets": len(tickets)}


class FaultRequest(FaultRequestBase):
    """Applied to the next `count` new creates or updates, in order. Reads are never faulted.

    error: nothing is applied. delay: the write waits `delay_seconds`, then is applied.
    accept_then_timeout: the write is applied, then the response waits `delay_seconds`.
    """

    mode: Literal["error", "delay", "accept_then_timeout"]


add_fault_routes(api, store.faults, FaultRequest)


@app.get("/health")
def health() -> dict[str, str]:
    """Public, for the docker healthcheck."""
    return {"status": "ok"}


# --- Tickets ----------------------------------------------------------------------------


class TicketCreate(BaseModel):
    local_ticket_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    subject: str = Field(min_length=1, max_length=200)
    status: Status = "open"


def same_ticket(ticket: Ticket, request: TicketCreate) -> bool:
    return (ticket.customer_id, ticket.subject) == (request.customer_id, request.subject)


def existing_ticket(request: TicketCreate) -> TicketOut | None:
    """The ticket for a repeated local_ticket_id, or None if it is new. Caller holds the lock."""
    existing = store.by_local_id.get(request.local_ticket_id)
    if existing is None:
        return None
    if not same_ticket(existing, request):
        raise MockError(
            status.HTTP_409_CONFLICT,
            "LOCAL_TICKET_ID_REUSED",
            "A ticket for this local_ticket_id exists with different details.",
        )
    return ticket_out(existing)


@api.post(
    "/tickets",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_200_OK: {"description": "Replay of an earlier request"}},
)
def create_ticket(request: TicketCreate, response: Response) -> TicketOut:
    with store.lock:
        replay = existing_ticket(request)
        if replay is not None:
            response.status_code = status.HTTP_200_OK
            return replay
        fault = take_write_fault()

    if fault is not None and fault.mode == "delay":
        hold_response(fault)
        fault = None

    with store.lock:
        # Another request may have created it while this one waited.
        replay = existing_ticket(request)
        if replay is not None:
            response.status_code = status.HTTP_200_OK
            return replay
        now = datetime.now(UTC)
        ticket = Ticket(
            ref=f"ext_{secrets.token_hex(8)}",
            local_ticket_id=request.local_ticket_id,
            customer_id=request.customer_id,
            subject=request.subject,
            status=request.status,
            created_at=now,
            updated_at=now,
        )
        store.add(ticket)
        result = ticket_out(ticket)

    hold_response(fault)
    return result


@api.get("/tickets/{ref}")
def get_ticket(ref: str) -> TicketOut:
    with store.lock:
        ticket = store.tickets.get(ref)
        if ticket is None:
            raise not_found()
        return ticket_out(ticket)


class TicketUpdate(BaseModel):
    # e.g. tku:<proposal_id>; the same ID is sent again on every retry of this update.
    update_id: str = Field(min_length=1, max_length=200)
    status: Status | None = None
    note: str | None = Field(default=None, min_length=1, max_length=4000)

    @model_validator(mode="after")
    def changes_something(self) -> "TicketUpdate":
        if self.status is None and self.note is None:
            raise ValueError("an update needs a status, a note, or both")
        return self


def same_update(update: Update, ref: str, request: TicketUpdate) -> bool:
    return (update.ticket_ref, update.status, update.note) == (ref, request.status, request.note)


def replay_or_conflict(ref: str, request: TicketUpdate) -> UpdateOut | None:
    """The stored result for a repeated update_id, or None if it is new. Caller holds the lock."""
    existing = store.updates.get(request.update_id)
    if existing is None:
        return None
    if not same_update(existing, ref, request):
        raise MockError(
            status.HTTP_409_CONFLICT,
            "UPDATE_ID_REUSED",
            "This update_id was already used with a different update.",
        )
    return UpdateOut(**existing.__dict__)


@api.post(
    "/tickets/{ref}/updates",
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_200_OK: {"description": "Replay of an earlier update"}},
)
def update_ticket(ref: str, request: TicketUpdate, response: Response) -> UpdateOut:
    """Change the status and/or add a note."""
    with store.lock:
        replay = replay_or_conflict(ref, request)
        if replay is not None:
            response.status_code = status.HTTP_200_OK
            return replay
        if ref not in store.tickets:
            raise not_found()
        fault = take_write_fault()

    if fault is not None and fault.mode == "delay":
        hold_response(fault)
        fault = None

    with store.lock:
        # A retry may have applied it while this one waited.
        replay = replay_or_conflict(ref, request)
        if replay is not None:
            response.status_code = status.HTTP_200_OK
            return replay
        ticket = store.tickets[ref]
        now = datetime.now(UTC)
        if request.status is not None:
            ticket.status = request.status
        if request.note is not None:
            ticket.notes.append(Note(text=request.note, at=now))
        ticket.updated_at = now
        update = Update(
            update_id=request.update_id,
            ticket_ref=ref,
            status=request.status,
            note=request.note,
            applied_at=now,
        )
        store.updates[update.update_id] = update
        result = UpdateOut(**update.__dict__)

    hold_response(fault)
    return result


app.include_router(api)
