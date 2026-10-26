"""Return read service (W1.12). Ownership of the order is checked first, in the service."""

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.schemas.returns import ReturnView
from app.services.errors import ServiceFailure
from app.services.orders import ensure_order_owner


def get_return_status(db: Session, customer: AuthenticatedCustomer, order_id: str) -> ReturnView:
    ensure_order_owner(db, customer, order_id)
    row = db.execute(
        text(
            "SELECT id AS return_id, order_id, status, requested_at, received_at, "
            f"inspection_status FROM {SCHEMA}.returns WHERE order_id = :order"
        ),
        {"order": order_id},
    ).one_or_none()
    if row is None:
        raise ServiceFailure("RETURN_NOT_FOUND", "No return has been started for this order.")
    return ReturnView.model_validate(row._mapping)
