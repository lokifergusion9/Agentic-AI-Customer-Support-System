"""Approval endpoints (W3.4): the reviewer's queue, and deciding one approval.

The reviewer is whoever the token says, with the role it carries. A decision is saved first,
then the paused run is resumed on the same thread: inline, or by a queued job with
`RUN_EXECUTION=worker` (W3.13, the answer is then 202 Accepted). The resumed workflow reads the
stored decision, so nothing in the request or the chat can authorize a refund (W3.7).
"""

from fastapi import APIRouter, Response, status
from fastapi.responses import JSONResponse

from app.api.runs import Db, Dispatch, accepted
from app.auth.reviewer import CurrentReviewer
from app.schemas.approvals import ApprovalDecisionRequest, ApprovalDecisionResult
from app.schemas.errors import ErrorResponse
from app.services.approvals import (
    PendingApprovalView,
    decide_approval,
    get_approval,
    list_pending_approvals,
)
from app.services.errors import ServiceFailure
from app.services.runs import get_run_for_reviewer

router = APIRouter(tags=["approvals"])

STATUS_FOR = {
    "APPROVAL_NOT_FOUND": status.HTTP_404_NOT_FOUND,
    "REVIEWER_NOT_PERMITTED": status.HTTP_403_FORBIDDEN,
    "APPROVAL_ALREADY_DECIDED": status.HTTP_409_CONFLICT,
    "APPROVAL_EXPIRED": status.HTTP_409_CONFLICT,
}
# The approval is closed either way; resuming lets a paused run follow the stored outcome.
RESUME_ON = {"APPROVAL_ALREADY_DECIDED", "APPROVAL_EXPIRED"}
ERRORS = {
    code: {"model": ErrorResponse}
    for code in (
        status.HTTP_403_FORBIDDEN,
        status.HTTP_404_NOT_FOUND,
        status.HTTP_409_CONFLICT,
        status.HTTP_503_SERVICE_UNAVAILABLE,
    )
}


def failure_response(failure: ServiceFailure) -> JSONResponse:
    return JSONResponse(
        status_code=STATUS_FOR.get(failure.error_code, status.HTTP_400_BAD_REQUEST),
        content=failure.to_response().model_dump(),
    )


@router.get("/approvals", response_model=list[PendingApprovalView], responses=ERRORS)
def list_approvals(reviewer: CurrentReviewer, db: Db) -> list[PendingApprovalView]:
    return list_pending_approvals(db, reviewer)


@router.post(
    "/approvals/{approval_id}/decision",
    response_model=ApprovalDecisionResult,
    responses={**ERRORS, 202: {"model": ApprovalDecisionResult}},
)
def decide(
    approval_id: str,
    body: ApprovalDecisionRequest,
    reviewer: CurrentReviewer,
    db: Db,
    dispatcher: Dispatch,
    response: Response,
) -> ApprovalDecisionResult | JSONResponse:
    try:
        decide_approval(db, reviewer, approval_id, body.decision)
    except ServiceFailure as failure:
        if failure.error_code in RESUME_ON:
            # E.g. a retried request after the first one saved the decision but not the resume.
            dispatcher.decision(db, approval_id)
        return failure_response(failure)
    run = dispatcher.decision(db, approval_id)
    accepted(response, dispatcher)
    return ApprovalDecisionResult(
        approval=get_approval(db, approval_id), run=get_run_for_reviewer(db, run.id)
    )
