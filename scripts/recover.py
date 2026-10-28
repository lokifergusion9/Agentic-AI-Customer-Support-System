"""Reconcile uncertain refunds and resume unfinished runs (W3.12).

    uv run python -m scripts.recover

Safe to run any number of times; the W3.13 worker runs the same sweep before taking jobs.
"""

import sys

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.runs import get_workflow_services
from app.db.session import get_engine
from app.workflows.recovery import recover


def main() -> int:
    try:
        services = get_workflow_services()
    except HTTPException as exc:
        print(f"Cannot build the workflow services: {exc.detail}", file=sys.stderr)
        return 1
    with Session(get_engine()) as db:
        report = recover(db, services)
    print(f"refund operations reconciled: {report.operations or 'none'}")
    print(f"approvals expired:            {report.expired_approvals or 'none'}")
    print(f"runs resumed:                 {report.runs or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
