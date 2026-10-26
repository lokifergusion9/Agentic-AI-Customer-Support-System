"""Mint and verify locally signed HS256 JWTs (Q5).

`sub` maps to `customers.external_auth_subject` (customers) or is the reviewer's identity.
`role` is one of ROLES and drives the customer / reviewer permission checks.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

from app.config import Settings, get_settings

ROLES = ("customer", "reviewer", "senior_reviewer")
REQUIRED_CLAIMS = ["exp", "iat", "sub", "role"]


class InvalidTokenError(Exception):
    """The token is missing, malformed, tampered, expired or has bad claims."""


@dataclass(frozen=True)
class TokenClaims:
    subject: str
    role: str
    expires_at: datetime


def mint_token(
    subject: str,
    role: str,
    *,
    expires_in: timedelta | None = None,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> str:
    """Sign a token for `subject` with `role`. Used by the dev script and tests."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}, expected one of {ROLES}")
    settings = settings or get_settings()
    issued_at = now or datetime.now(UTC)
    expires_in = expires_in or timedelta(minutes=settings.jwt_expiry_minutes)
    claims = {"sub": subject, "role": role, "iat": issued_at, "exp": issued_at + expires_in}
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, *, settings: Settings | None = None) -> TokenClaims:
    """Verify the signature, algorithm, expiry and required claims, or raise InvalidTokenError."""
    settings = settings or get_settings()
    try:
        # Only the configured algorithm is accepted, so `alg: none` or a swapped
        # algorithm in the token header is rejected.
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": REQUIRED_CLAIMS},
        )
    except jwt.PyJWTError as error:
        raise InvalidTokenError(str(error)) from error

    subject, role = claims["sub"], claims["role"]
    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError("sub must be a non-empty string")
    if role not in ROLES:
        raise InvalidTokenError(f"unknown role {role!r}")
    return TokenClaims(
        subject=subject,
        role=role,
        expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
    )
