"""Mint a demo JWT for a seeded customer or reviewer (Q5, W1.5).

    uv run python -m scripts.mint_token auth|alice
    uv run python -m scripts.mint_token auth|reviewer_riya --role reviewer --minutes 120

Signed with JWT_SECRET from `.env`. Send it as `Authorization: Bearer <token>`.
"""

import argparse
from datetime import timedelta

from app.auth.tokens import ROLES, mint_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("subject", help="external auth subject, e.g. auth|alice")
    parser.add_argument("--role", choices=ROLES, default="customer")
    parser.add_argument("--minutes", type=int, help="lifetime (default: JWT_EXPIRY_MINUTES)")
    args = parser.parse_args()

    expires_in = timedelta(minutes=args.minutes) if args.minutes else None
    print(mint_token(args.subject, args.role, expires_in=expires_in))


if __name__ == "__main__":
    main()
