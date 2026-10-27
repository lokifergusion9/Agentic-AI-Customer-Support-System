"""Run endpoints (W2.14): start a resolution, read it, its timeline, and reply to a question.

Runs execute inline for now (Q7): POST returns when the workflow finishes or pauses for the
customer or a reviewer. W3.13 switches to 202 Accepted with a worker.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.agents.config import AgentConfig, build_provider
from app.auth.customer import CurrentCustomer, get_current_customer
from app.auth.dependencies import get_token_claims
from app.auth.reviewer import REVIEWER_ROLES
from app.auth.tokens import TokenClaims
from app.db.session import get_db
from app.policies.embeddings import get_embedder
from app.schemas.errors import ErrorResponse
from app.schemas.runs import RunEventView, RunMessage
from app.services import runs
from app.services.errors import ServiceFailure
from app.services.providers import get_payments_client, get_shipping_client
from app.services.runs import RunView
from app.workflows.checkpoints import get_checkpointer
from app.workflows.runner import WorkflowServices, resume_with_message, start_resolution

router = APIRouter(tags=["runs"])

STATUS_FOR = {
    "TICKET_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "RUN_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "RUN_ALREADY_ACTIVE": status.HTTP_409_CONFLICT,
    "RUN_NOT_WAITING": status.HTTP_409_CONFLICT,
}
ERRORS = {
    code: {"model": ErrorResponse}
    for code in (status.HTTP_404_NOT_FOUND, status.HTTP_409_CONFLICT, 503)
}


def get_workflow_services() -> WorkflowServices:
    config = AgentConfig.from_settings()
    try:
        provider = build_provider(config)
    except ValueError as exc:
        # LLM_MODEL or LLM_API_KEY is not configured: say so instead of failing mid-run.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from None
    return WorkflowServices(
        provider=provider,
        config=config,
        payments=get_payments_client(),
        shipping=get_shipping_client(),
        embedder=get_embedder(),
        checkpointer=get_checkpointer(),
    )


Services = Annotated[WorkflowServices, Depends(get_workflow_services)]
Db = Annotated[Session, Depends(get_db)]


def failure_response(failure: ServiceFailure) -> JSONResponse:
    return JSONResponse(
        status_code=STATUS_FOR.get(failure.error_code, status.HTTP_400_BAD_REQUEST),
        content=failure.to_response().model_dump(),
    )


@router.post(
    "/tickets/{ticket_id}/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=RunView,
    responses=ERRORS,
)
def start_run(
    ticket_id: str, customer: CurrentCustomer, db: Db, services: Services
) -> RunView | JSONResponse:
    try:
        return start_resolution(db, customer, ticket_id, services)
    except ServiceFailure as failure:
        return failure_response(failure)


@router.get("/runs/{run_id}", response_model=RunView, responses=ERRORS)
def get_run(run_id: str, customer: CurrentCustomer, db: Db) -> RunView | JSONResponse:
    try:
        return runs.get_run(db, customer, run_id)
    except ServiceFailure as failure:
        return failure_response(failure)


@router.get("/runs/{run_id}/events", response_model=list[RunEventView], responses=ERRORS)
def get_run_events(
    run_id: str, claims: Annotated[TokenClaims, Depends(get_token_claims)], db: Db
) -> list[RunEventView] | JSONResponse:
    """The run's customer or any reviewer; everyone else gets the same 404."""
    try:
        if claims.role in REVIEWER_ROLES:
            runs.get_run_for_reviewer(db, run_id)
        else:
            runs.get_run(db, get_current_customer(claims, db), run_id)
    except ServiceFailure as failure:
        return failure_response(failure)
    return [RunEventView.model_validate(event) for event in runs.list_run_events(db, run_id)]


@router.post("/runs/{run_id}/messages", response_model=RunView, responses=ERRORS)
def post_run_message(
    run_id: str, body: RunMessage, customer: CurrentCustomer, db: Db, services: Services
) -> RunView | JSONResponse:
    try:
        return resume_with_message(db, customer, run_id, body.message, services)
    except ServiceFailure as failure:
        return failure_response(failure)
