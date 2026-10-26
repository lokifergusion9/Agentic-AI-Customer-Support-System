"""POST /tickets: a customer opens a support ticket (W1.7)."""

from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.auth.customer import CurrentCustomer
from app.db.session import get_db
from app.schemas.errors import ErrorResponse
from app.schemas.tickets import TicketCreate, TicketResponse
from app.services import tickets
from app.services.errors import ServiceFailure

router = APIRouter(tags=["tickets"])


@router.post(
    "/tickets",
    status_code=status.HTTP_201_CREATED,
    response_model=TicketResponse,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
def create_ticket(
    request: TicketCreate,
    customer: CurrentCustomer,
    db: Annotated[Session, Depends(get_db)],
) -> TicketResponse | JSONResponse:
    try:
        return tickets.create_ticket(db, customer, request)
    except ServiceFailure as failure:
        # ORDER_NOT_FOUND is the only failure create_ticket raises.
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content=failure.to_response().model_dump()
        )
