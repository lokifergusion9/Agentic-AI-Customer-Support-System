"""Token minting and verification reject every malformed or forged token (W1.5)."""

import base64
import json
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.auth.tokens import InvalidTokenError, decode_token, mint_token
from app.config import Settings

# 64 bytes, long enough for HS512 too, so PyJWT does not warn about key length.
SETTINGS = Settings(jwt_secret="unit-test-secret-" + "x" * 47)
OTHER_SETTINGS = Settings(jwt_secret="another-test-secret-" + "y" * 44)


def claims(**overrides: object) -> dict[str, object]:
    now = datetime.now(UTC)
    base: dict[str, object] = {
        "sub": "auth|alice",
        "role": "customer",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    base.update(overrides)
    return {key: value for key, value in base.items() if value is not None}


def sign(payload: dict[str, object], algorithm: str = "HS256") -> str:
    return jwt.encode(payload, SETTINGS.jwt_secret, algorithm=algorithm)


def test_round_trip() -> None:
    token = mint_token("auth|alice", "customer", settings=SETTINGS)
    decoded = decode_token(token, settings=SETTINGS)
    assert (decoded.subject, decoded.role) == ("auth|alice", "customer")
    assert decoded.expires_at > datetime.now(UTC)


def test_expiry_defaults_to_settings() -> None:
    now = datetime(2026, 10, 25, 10, tzinfo=UTC)
    token = mint_token("auth|alice", "customer", now=now, settings=SETTINGS)
    payload = jwt.decode(
        token, SETTINGS.jwt_secret, algorithms=["HS256"], options={"verify_exp": False}
    )
    assert payload["exp"] - payload["iat"] == SETTINGS.jwt_expiry_minutes * 60


def test_mint_rejects_unknown_role() -> None:
    with pytest.raises(ValueError):
        mint_token("auth|alice", "admin", settings=SETTINGS)


def test_expired_token_is_rejected() -> None:
    issued = datetime.now(UTC) - timedelta(hours=2)
    token = mint_token(
        "auth|alice", "customer", now=issued, expires_in=timedelta(hours=1), settings=SETTINGS
    )
    with pytest.raises(InvalidTokenError):
        decode_token(token, settings=SETTINGS)


def test_wrong_secret_is_rejected() -> None:
    token = mint_token("auth|alice", "customer", settings=OTHER_SETTINGS)
    with pytest.raises(InvalidTokenError):
        decode_token(token, settings=SETTINGS)


def test_tampered_payload_is_rejected() -> None:
    header, payload, signature = mint_token("auth|alice", "customer", settings=SETTINGS).split(".")
    original = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    # Escalate the role but keep the original signature.
    forged = base64.urlsafe_b64encode(json.dumps({**original, "role": "senior_reviewer"}).encode())
    token = ".".join([header, forged.decode().rstrip("="), signature])
    with pytest.raises(InvalidTokenError):
        decode_token(token, settings=SETTINGS)


def test_alg_none_is_rejected() -> None:
    token = jwt.encode(claims(), None, algorithm="none")
    with pytest.raises(InvalidTokenError):
        decode_token(token, settings=SETTINGS)


def test_other_algorithm_is_rejected() -> None:
    token = sign(claims(), algorithm="HS512")
    with pytest.raises(InvalidTokenError):
        decode_token(token, settings=SETTINGS)


@pytest.mark.parametrize("missing", ["sub", "role", "iat", "exp"])
def test_missing_claim_is_rejected(missing: str) -> None:
    token = sign(claims(**{missing: None}))
    with pytest.raises(InvalidTokenError):
        decode_token(token, settings=SETTINGS)


@pytest.mark.parametrize("role", ["admin", "", 1])
def test_unknown_role_is_rejected(role: object) -> None:
    with pytest.raises(InvalidTokenError):
        decode_token(sign(claims(role=role)), settings=SETTINGS)


def test_empty_subject_is_rejected() -> None:
    with pytest.raises(InvalidTokenError):
        decode_token(sign(claims(sub="")), settings=SETTINGS)


@pytest.mark.parametrize("token", ["", "abc", "a.b.c"])
def test_garbage_is_rejected(token: str) -> None:
    with pytest.raises(InvalidTokenError):
        decode_token(token, settings=SETTINGS)
