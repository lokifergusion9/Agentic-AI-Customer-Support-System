"""Shared FastAPI dependency: verified token claims from the `Authorization: Bearer` header."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.tokens import InvalidTokenError, TokenClaims, decode_token

# auto_error=False so a missing header gets the same 401 as a bad token.
bearer_scheme = HTTPBearer(auto_error=False)


def unauthorized(detail: str = "Invalid or missing credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_token_claims(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> TokenClaims:
    if credentials is None:
        raise unauthorized()
    try:
        return decode_token(credentials.credentials)
    except InvalidTokenError:
        # The reason stays server-side; clients only learn that the token was rejected.
        raise unauthorized() from None
