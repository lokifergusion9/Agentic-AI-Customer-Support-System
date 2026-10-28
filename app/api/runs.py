"""Run endpoints (W2.14): start a resolution, read it, its timeline, and reply to a question.

With `RUN_EXECUTION=inline` a POST returns when the workflow finishes or pauses for the
customer or a reviewer. With `RUN_EXECUTION=worker` (W3.13) it records the request, queues a
job and answers 202 Accepted with the run; clients poll `GET /runs/{run_id}`.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.agents.config import AgentConfig, build_provider
from app.auth.customer import CurrentCustomer, get_current_customer
from app.auth.dependencies import get_token_claims
from app.auth.reviewer import REVIEWER_ROLES
from app.auth.tokens import TokenClaims
from app.config import get_settings
from app.db.session import get_db
from app.policies.embeddings import get_embedder
from app.schemas.errors import ErrorResponse
from app.schemas.runs import RunEventView, RunMessage
from app.services import runs
from app.services.errors import ServiceFailure
from app.services.providers import get_payments_client, get_shipping_client
from app.services.runs import RunView
from app.workflows.checkpoints import get_checkpointer
from app.workflows.dispatch import Dispatcher, InlineDispatcher, WorkerDispatcher
from app.workflows.runner import WorkflowServices

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


def get_dispatcher() -> Dispatcher:
    """Worker mode queues jobs and needs no model credentials; inline mode builds the services."""
    if get_settings().run_execution == "worker":
        return WorkerDispatcher(AgentConfig.from_settings())
    return InlineDispatcher(get_workflow_services())


Dispatch = Annotated[Dispatcher, Depends(get_dispatcher)]
Db = Annotated[Session, Depends(get_db)]


def accepted(response: Response, dispatcher: Dispatcher) -> None:
    if dispatcher.accepted:
        response.status_code = status.HTTP_202_ACCEPTED


def failure_response(failure: ServiceFailure) -> JSONResponse:
    return JSONResponse(
        status_code=STATUS_FOR.get(failure.error_code, status.HTTP_400_BAD_REQUEST),
        content=failure.to_response().model_dump(),
    )


@router.post(
    "/tickets/{ticket_id}/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=RunView,
    responses={**ERRORS, 202: {"model": RunView}},
)
def start_run(
    ticket_id: str, customer: CurrentCustomer, db: Db, dispatcher: Dispatch, response: Response
) -> RunView | JSONResponse:
    try:
        run = dispatcher.start(db, customer, ticket_id)
    except ServiceFailure as failure:
        return failure_response(failure)
    accepted(response, dispatcher)
    return run


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


@router.post(
    "/runs/{run_id}/messages",
    response_model=RunView,
    responses={**ERRORS, 202: {"model": RunView}},
)
def post_run_message(
    run_id: str,
    body: RunMessage,
    customer: CurrentCustomer,
    db: Db,
    dispatcher: Dispatch,
    response: Response,
) -> RunView | JSONResponse:
    try:
        run = dispatcher.message(db, customer, run_id, body.message)
    except ServiceFailure as failure:
        return failure_response(failure)
    accepted(response, dispatcher)
    return run
