"""Request and response bodies for the approval endpoints (W3.4)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.services.approvals import ApprovalView
from app.services.runs import RunView


class ApprovalDecisionRequest(BaseModel):
    """POST /approvals/{approval_id}/decision. The reviewer comes from the token, never the body."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected"]


class ApprovalDecisionResult(BaseModel):
    """The stored decision, and the run as it stands after resuming on it."""

    approval: ApprovalView
    run: RunView
