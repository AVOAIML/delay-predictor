"""LOCAL DEVELOPMENT ONLY — make the Configurator API callable on a laptop.

`services/configurator/security.py` authenticates every `/api/*` request against
the MAIN MaXXFlow product's identity schema: `public.users` (with `upn` and
`is_ci_admin`), `public.tenants`, `public.user_tenants`, and a `public.master_data`
table holding the status codes. This repository's `maxxflow db-provision` does
not create any of that — `schema_def.PUBLIC_TABLES` declares only a minimal
`users` and `tenants` — because in a real deployment the API points at the
product's database, where those tables already exist.

So on a fresh local stack every `/api/*` call fails with "Access token
required", and would still fail with a token because the tables the verifier
reads are absent. This script closes exactly that gap and nothing else:

  1. creates the missing identity objects IF THEY DO NOT EXIST,
  2. seeds one active CI-admin user and one active tenant,
  3. mints a short-lived HS256 token for that user.

It is deliberately NOT part of `db-provision`: inventing the product's identity
schema in the shared DDL would let a mistake here propagate to a real
environment. Everything it creates is additive and idempotent, and it only ever
touches `public`.

    # start the API with a matching secret, then:
    uv run python scripts/dev_auth.py --tenant demo
    # -> prints an Authorization header you can paste into Postman

Never run this against a database you did not create yourself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import uuid

import jwt
import sqlalchemy as sa

DEFAULT_SECRET = "local-dev-jwt-secret-change-me-32b"  # >= 32 bytes: HS256 minimum
DEFAULT_EMAIL = "dev@maxxflow.local"

_NS = uuid.UUID("9c3f21b7-4d6a-4e58-9f02-5b7c8d1e3a44")

ACTIVE_USER = "USER_STATUS_ACTIVE"
ACTIVE_TENANT = "TENANT_STATUS_ACTIVE"
ACTIVE_MEMBERSHIP = "USER_TENANT_STATUS_ACTIVE"


def _u(*parts: str) -> str:
    return str(uuid.uuid5(_NS, ":".join(parts)))


#: One statement per entry — kept as a list rather than one script so nothing
#: has to parse SQL to split it, and every statement is additive and repeatable.
_DDL: tuple[str, ...] = (
    # The status vocabulary the verifier joins against. In the product this is
    # a far richer table; only id and code are read here.
    """
    CREATE TABLE IF NOT EXISTS public.master_data (
        id          uuid PRIMARY KEY,
        category_id uuid,
        name        varchar(255),
        code        varchar(255),
        status      varchar(10) NOT NULL DEFAULT 'active',
        created_at  timestamp NOT NULL DEFAULT now(),
        updated_at  timestamp,
        deleted_at  timestamp
    )
    """,
    # Columns security.py reads that schema_def's minimal `users` does not
    # declare.
    "ALTER TABLE public.users ADD COLUMN IF NOT EXISTS upn varchar(255)",
    "ALTER TABLE public.users ADD COLUMN IF NOT EXISTS is_ci_admin boolean NOT NULL DEFAULT false",
    # Membership. Queried even for a CI admin (whose membership may legitimately
    # be absent), so the TABLE must exist or the lookup raises instead of
    # returning nothing.
    """
    CREATE TABLE IF NOT EXISTS public.user_tenants (
        id         uuid PRIMARY KEY,
        user_id    uuid NOT NULL,
        tenant_id  uuid NOT NULL,
        role_ids   uuid[] NOT NULL DEFAULT '{}',
        status_id  uuid NOT NULL,
        created_at timestamp NOT NULL DEFAULT now(),
        updated_at timestamp,
        deleted_at timestamp
    )
    """,
)


def _upsert_status(conn, code: str) -> str:
    status_id = _u("status", code)
    conn.execute(
        sa.text(
            "INSERT INTO public.master_data (id, name, code, status) "
            "VALUES (:id, :code, :code, 'active') ON CONFLICT (id) DO NOTHING"
        ),
        {"id": status_id, "code": code},
    )
    return status_id


def bootstrap(tenant: str, email: str) -> dict:
    from maxxflow_core.settings import get_settings
    from maxxflow_data.engine import get_data_access

    settings = get_settings()
    schema_name = settings.tenant_schema(tenant)
    engine = get_data_access().engine

    with engine.begin() as conn:
        for statement in _DDL:
            conn.exec_driver_sql(statement)

        user_status = _upsert_status(conn, ACTIVE_USER)
        tenant_status = _upsert_status(conn, ACTIVE_TENANT)
        membership_status = _upsert_status(conn, ACTIVE_MEMBERSHIP)

        user_id = _u("user", email)
        conn.execute(
            sa.text(
                "INSERT INTO public.users (id, email, name, auth_provider, status_id, upn, "
                "is_ci_admin) VALUES (:id, :email, 'Local Developer', 'local', :status, :email, "
                "true) ON CONFLICT (id) DO UPDATE SET is_ci_admin = true, "
                "status_id = EXCLUDED.status_id"
            ),
            {"id": user_id, "email": email, "status": user_status},
        )

        tenant_id = _u("tenant", tenant)
        conn.execute(
            sa.text(
                "INSERT INTO public.tenants (id, slug, name, schema_name, status_id) "
                "VALUES (:id, :slug, :name, :schema, :status) "
                "ON CONFLICT (id) DO UPDATE SET status_id = EXCLUDED.status_id, "
                "schema_name = EXCLUDED.schema_name"
            ),
            {
                "id": tenant_id,
                "slug": tenant,
                "name": f"{tenant} (local)",
                "schema": schema_name,
                "status": tenant_status,
            },
        )

        # An explicit active membership as well as the CI-admin flag, so the
        # same setup works if that flag is ever tightened.
        conn.execute(
            sa.text(
                "INSERT INTO public.user_tenants (id, user_id, tenant_id, role_ids, status_id) "
                "VALUES (:id, :user, :tenant, '{}', :status) "
                "ON CONFLICT (id) DO UPDATE SET status_id = EXCLUDED.status_id"
            ),
            {
                "id": _u("membership", email, tenant),
                "user": user_id,
                "tenant": tenant_id,
                "status": membership_status,
            },
        )

    return {"user_id": user_id, "tenant_id": tenant_id, "schema_name": schema_name}


def mint_token(email: str, secret: str, hours: int) -> str:
    """An HS256 token the API's JWTVerifier accepts when JWT_SECRET matches.

    `exp` is required by the verifier, and `email` is the claim it resolves the
    user by.
    """
    now = dt.datetime.now(tz=dt.timezone.utc)
    return jwt.encode(
        {
            "email": email,
            "preferred_username": email,
            "iat": now,
            "exp": now + dt.timedelta(hours=hours),
        },
        secret,
        algorithm="HS256",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dev_auth",
        description="Seed local identity rows and mint a Configurator API token (LOCAL ONLY).",
    )
    parser.add_argument("--tenant", default="demo")
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument(
        "--secret",
        default=os.getenv("JWT_SECRET") or DEFAULT_SECRET,
        help="must match the JWT_SECRET the API process was started with",
    )
    parser.add_argument("--hours", type=int, default=12, help="token lifetime")
    parser.add_argument(
        "--token-only", action="store_true", help="print just the token, nothing else"
    )
    args = parser.parse_args(argv)

    bootstrap(args.tenant, args.email)
    token = mint_token(args.email, args.secret, args.hours)

    if args.token_only:
        print(token)
        return 0

    print("Local identity seeded. Start the API with the SAME secret:")
    print(f"  JWT_SECRET={args.secret}")
    print()
    print("Postman / curl headers:")
    print(f"  Authorization: Bearer {token}")
    print(f"  x-tenant-slug: {args.tenant}")
    print()
    print(f"Token valid for {args.hours}h. Re-run this script to mint a fresh one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
