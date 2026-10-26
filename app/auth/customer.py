"""Customer authentication (W1.5).

The customer is always derived from the verified token: `sub` is looked up in
`customers.external_auth_subject`. Endpoints and services take the customer from this
dependency, never from the request body or from the model.
"""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.dependencies import get_token_claims, unauthorized
from app.auth.tokens import TokenClaims
from app.db import SCHEMA
from app.db.session import get_db


@dataclass(frozen=True)
class AuthenticatedCustomer:
    customer_id: str
    subject: str


def get_current_customer(
    claims: Annotated[TokenClaims, Depends(get_token_claims)],
    db: Annotated[Session, Depends(get_db)],
) -> AuthenticatedCustomer:
    if claims.role != "customer":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Customer role required")

    customer_id = db.execute(
        text(f"SELECT id FROM {SCHEMA}.customers WHERE external_auth_subject = :subject"),
        {"subject": claims.subject},
    ).scalar_one_or_none()
    if customer_id is None:
        # Same 401 as a bad token, so the response does not reveal which subjects exist.
        raise unauthorized()
    return AuthenticatedCustomer(customer_id=customer_id, subject=claims.subject)


CurrentCustomer = Annotated[AuthenticatedCustomer, Depends(get_current_customer)]
