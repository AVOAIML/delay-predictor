"""Authentication and tenant-scoped authorization for the Configurator API.

Tokens are verified before any claim is trusted.  User status, tenant status,
membership, roles and permissions remain database-backed, matching the main API's
authorization model instead of trusting caller-supplied role claims.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
import sqlalchemy as sa
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError

from maxxflow_core.settings import get_settings

ACTIVE_USER = "USER_STATUS_ACTIVE"
ACTIVE_MEMBERSHIP = "USER_TENANT_STATUS_ACTIVE"
ALLOWED_TENANT_STATUSES = {"TENANT_STATUS_ACTIVE", "TENANT_STATUS_CANCEL"}
ADMIN_ROLES = frozenset({"ci-admin", "maxxflow-admin"})

TRAIN_PERMISSION = "ml-models:train"
PUBLISH_PERMISSION = "ml-models:publish"
ROLLBACK_PERMISSION = "ml-models:rollback"
DELETE_PERMISSION = "ml-models:delete"

_SAFE_SCHEMA = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TENANT_PATH = re.compile(r"^/api/([^/]+)/(?:models(?:/|$)|inventory-dashboard(?:/|$))")
_SENSITIVE_PATHS = (
    (re.compile(r"^/api/[^/]+/models/[^/]+/train/?$"), "POST", TRAIN_PERMISSION),
    (re.compile(r"^/api/[^/]+/models/[^/]+/publish/?$"), "POST", PUBLISH_PERMISSION),
    (re.compile(r"^/api/[^/]+/models/[^/]+/rollback/?$"), "POST", ROLLBACK_PERMISSION),
    (re.compile(r"^/api/[^/]+/models(?:/[^/]+)?/?$"), "DELETE", DELETE_PERMISSION),
    (re.compile(r"^/api/train/[^/]+/?$"), "GET", TRAIN_PERMISSION),
)


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    email: str
    tenant_id: str
    tenant_slug: str
    roles: frozenset[str]
    permissions: frozenset[str]
    is_ci_admin: bool = False

    def has_permission(self, permission: str) -> bool:
        return (
            self.is_ci_admin
            or bool(self.roles & ADMIN_ROLES)
            or permission in self.permissions
            or "ml-models:*" in self.permissions
            or "*:*" in self.permissions
        )


def _unauthorized(detail: str = "Invalid or expired bearer token") -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _oidc_metadata(authority: str) -> tuple[str, str]:
    base = authority.rstrip("/")
    last_error: Exception | None = None
    for suffix in (
        "/v2.0/.well-known/openid-configuration",
        "/.well-known/openid-configuration",
    ):
        try:
            with urllib.request.urlopen(base + suffix, timeout=10) as response:  # noqa: S310
                document = json.loads(response.read())
            if document.get("jwks_uri") and document.get("issuer"):
                return str(document["jwks_uri"]), str(document["issuer"])
        except Exception as exc:  # try the authority's other discovery layout
            last_error = exc
    raise InvalidTokenError(f"OIDC discovery failed: {last_error}")


class JWTVerifier:
    """Verify backend HS256 tokens and legacy Entra RS256 tokens, fail closed."""

    def __init__(self) -> None:
        self.secret = os.getenv("JWT_SECRET", "").strip()
        self.audience = (
            os.getenv("JWT_AUDIENCE")
            or os.getenv("AZURE_ENTRA_AUDIENCE")
            or os.getenv("AZURE_ENTRA_CLIENT_ID")
            or os.getenv("ENTRA_EXTERNAL_CLIENT_ID")
        )
        self.authority = (
            os.getenv("JWT_AUTHORITY")
            or os.getenv("AZURE_ENTRA_AUTHORITY")
            or os.getenv("ENTRA_EXTERNAL_AUTHORITY")
        )
        self.jwks_uri = os.getenv("JWT_JWKS_URI") or os.getenv("AZURE_ENTRA_JWKS_URI")
        self.issuer = os.getenv("JWT_ISSUER") or os.getenv("AZURE_ENTRA_ISSUER")
        self._jwks_client: PyJWKClient | None = None

    def _rs256_config(self) -> tuple[PyJWKClient, str] | None:
        if not self.jwks_uri and not self.authority:
            return None
        if not self.jwks_uri or not self.issuer:
            discovered_jwks, discovered_issuer = _oidc_metadata(self.authority or "")
            self.jwks_uri = self.jwks_uri or discovered_jwks
            self.issuer = self.issuer or discovered_issuer
        if self._jwks_client is None:
            self._jwks_client = PyJWKClient(self.jwks_uri, cache_keys=True, lifespan=600)
        return self._jwks_client, self.issuer

    def verify(self, token: str) -> dict[str, Any]:
        if token.count(".") != 2:
            raise _unauthorized()

        last_error: Exception | None = None
        try:
            rs256 = self._rs256_config()
        except Exception as exc:
            rs256 = None
            last_error = exc
        if rs256 is not None:
            client, issuer = rs256
            try:
                key = client.get_signing_key_from_jwt(token)
                return jwt.decode(
                    token,
                    key.key,
                    algorithms=["RS256"],
                    audience=self.audience,
                    issuer=issuer,
                    leeway=60,
                    options={"verify_aud": bool(self.audience), "require": ["exp"]},
                )
            except Exception as exc:
                last_error = exc

        if self.secret:
            try:
                return jwt.decode(
                    token,
                    self.secret,
                    algorithms=["HS256"],
                    leeway=60,
                    options={"verify_aud": False, "require": ["exp"]},
                )
            except Exception as exc:
                last_error = exc

        if last_error is None:
            last_error = InvalidTokenError("JWT verification is not configured")
        raise _unauthorized() from last_error


class AuthRepository:
    """Resolve identity, tenant membership and RBAC grants from PostgreSQL."""

    def __init__(self) -> None:
        database_url = get_settings().data_db_url
        if not database_url:
            raise RuntimeError("DATA_DB_URL is required for Configurator authorization")
        self.engine = sa.create_engine(database_url, pool_pre_ping=True)

    @staticmethod
    def _identifier(claims: dict[str, Any]) -> str:
        for name in ("email", "preferred_username", "unique_name", "upn"):
            value = claims.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise _unauthorized("Invalid bearer token: missing user identity claim")

    def authorize(self, claims: dict[str, Any], tenant_slug: str) -> AuthContext:
        identifier = self._identifier(claims)
        with self.engine.connect() as connection:
            user = connection.execute(
                sa.text(
                    """
                    SELECT u.id, u.email, u.is_ci_admin, us.code
                    FROM public.users u
                    JOIN public.master_data us ON us.id = u.status_id
                    WHERE u.deleted_at IS NULL
                      AND (LOWER(u.email) = LOWER(:identifier) OR u.upn = :identifier)
                    ORDER BY CASE WHEN LOWER(u.email) = LOWER(:identifier) THEN 0 ELSE 1 END
                    LIMIT 1
                    """
                ),
                {"identifier": identifier},
            ).mappings().first()
            if not user or user["code"] != ACTIVE_USER:
                raise _unauthorized("User for bearer token was not found or is inactive")

            tenant = connection.execute(
                sa.text(
                    """
                    SELECT t.id, t.slug, t.schema_name, ts.code
                    FROM public.tenants t
                    JOIN public.master_data ts ON ts.id = t.status_id
                    WHERE t.slug = :slug AND t.deleted_at IS NULL
                    LIMIT 1
                    """
                ),
                {"slug": tenant_slug},
            ).mappings().first()
            if not tenant or tenant["code"] not in ALLOWED_TENANT_STATUSES:
                raise HTTPException(403, "Caller is not authorized for the requested tenant")

            membership = connection.execute(
                sa.text(
                    """
                    SELECT ut.role_ids, ms.code
                    FROM public.user_tenants ut
                    JOIN public.master_data ms ON ms.id = ut.status_id
                    WHERE ut.user_id = :user_id AND ut.tenant_id = :tenant_id
                      AND ut.deleted_at IS NULL
                    LIMIT 1
                    """
                ),
                {"user_id": user["id"], "tenant_id": tenant["id"]},
            ).mappings().first()
            is_ci_admin = bool(user["is_ci_admin"])
            if (
                not is_ci_admin
                and (not membership or membership["code"] != ACTIVE_MEMBERSHIP)
            ):
                raise HTTPException(403, "Caller is not authorized for the requested tenant")

            role_ids = list(membership["role_ids"] or []) if membership else []
            roles: set[str] = set()
            permissions: set[str] = set()
            if role_ids:
                schema = str(tenant["schema_name"])
                if not _SAFE_SCHEMA.fullmatch(schema):
                    raise RuntimeError("Tenant has an invalid schema name")
                connection.exec_driver_sql(f'SET search_path TO "{schema}", public')
                rows = connection.execute(
                    sa.text(
                        """
                        SELECT DISTINCT r.code, p.resource, p.action
                        FROM roles r
                        LEFT JOIN role_permission rp ON rp.role_id = r.id
                        LEFT JOIN permissions p ON p.id = rp.permission_id
                        WHERE r.id = ANY(CAST(:role_ids AS uuid[]))
                          AND r.deleted_at IS NULL
                        """
                    ),
                    {"role_ids": role_ids},
                ).mappings()
                for row in rows:
                    roles.add(str(row["code"]))
                    if row["resource"] and row["action"]:
                        permissions.add(f'{row["resource"]}:{row["action"]}')

        return AuthContext(
            user_id=str(user["id"]),
            email=str(user["email"]),
            tenant_id=str(tenant["id"]),
            tenant_slug=str(tenant["slug"]),
            roles=frozenset(roles),
            permissions=frozenset(permissions),
            is_ci_admin=is_ci_admin,
        )


@lru_cache(maxsize=1)
def get_jwt_verifier() -> JWTVerifier:
    return JWTVerifier()


@lru_cache(maxsize=1)
def get_auth_repository() -> AuthRepository:
    return AuthRepository()


def _bearer_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Bearer token required")
    return token.strip()


def _required_permission(request: Request) -> str | None:
    for path_pattern, method, permission in _SENSITIVE_PATHS:
        if request.method == method and path_pattern.fullmatch(request.url.path):
            return permission
    return None


def authenticate_request(request: Request) -> AuthContext:
    """Authenticate one /api request and bind it to its requested tenant."""
    claims = get_jwt_verifier().verify(_bearer_token(request))
    tenant_slug = request.headers.get("x-tenant-slug", "").strip()
    if not tenant_slug:
        raise HTTPException(403, "x-tenant-slug is required")
    context = get_auth_repository().authorize(claims, tenant_slug)

    path_match = _TENANT_PATH.match(request.url.path)
    if path_match:
        path_tenant = path_match.group(1)
        # `global` is the shared base-model owner, not a database tenant. Members
        # may read it; mutation still passes through the admin/permission gate.
        if path_tenant != context.tenant_slug and path_tenant != "global":
            raise HTTPException(403, "Path tenant does not match x-tenant-slug")

    permission = _required_permission(request)
    if permission and not context.has_permission(permission):
        raise HTTPException(403, f"Missing required permission: {permission}")
    return context


def install_security(app: FastAPI) -> None:
    """Install the single authentication/authorization boundary for `/api/*`."""

    @app.middleware("http")
    async def secure_api(request: Request, call_next):
        # Browsers must be able to perform CORS preflight before credentials are
        # evaluated. The subsequent real request is still always authenticated.
        if request.url.path.startswith("/api/") and request.method != "OPTIONS":
            try:
                request.state.auth = authenticate_request(request)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers=exc.headers,
                )
        return await call_next(request)
