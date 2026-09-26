"""The tenant-path guard covers every tenant-scoped route.

`authenticate_request` authorizes a caller for the tenant in their
`x-tenant-slug` header, then separately checks that the tenant in the URL is
the same one — but only for paths matching `security._TENANT_PATH`, a regex
enumerating path prefixes. A route that reads or writes tenant data and is not
in that pattern authorizes against one tenant and then queries another, which
is a cross-tenant read.

Enumerating the prefixes by hand is the kind of thing that is correct when
written and wrong three routes later, so this asserts the property directly
against the app's own route table rather than against a list someone has to
remember to update.
"""

from __future__ import annotations

import pytest

from services.configurator.app import app
from services.configurator.security import _TENANT_PATH

#: Routes under /api/ that are deliberately NOT tenant-scoped, with the reason.
#: Anything else carrying a {tenant} segment must be covered by the guard.
_NOT_TENANT_SCOPED = {
    "/api/train/{run_id}": "keyed by run id; its own permission gate covers it",
}


def _tenant_routes() -> list[str]:
    return sorted(
        {
            route.path
            for route in app.routes
            if getattr(route, "path", "").startswith("/api/{tenant}/")
        }
    )


def test_the_app_actually_has_tenant_routes():
    # Guards the guard: if the route table stops matching this shape, the test
    # below would pass vacuously.
    assert len(_tenant_routes()) >= 5


@pytest.mark.parametrize("path", _tenant_routes())
def test_every_tenant_scoped_route_is_covered_by_the_guard(path):
    concrete = path.replace("{tenant}", "demo").replace("{key}", "m1_quote_line_win")
    assert _TENANT_PATH.match(concrete), (
        f"{path} takes a tenant in the URL but is not matched by security._TENANT_PATH, "
        "so a caller authorized for one tenant could read another by changing the URL. "
        "Add its prefix to that pattern."
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/demo/models",
        "/api/demo/models/m2_inventory/batch-predict",
        "/api/demo/inventory-dashboard",
        "/api/demo/delay-insights",
        "/api/demo/delay-insights/",
    ],
)
def test_known_tenant_paths_are_matched(path):
    match = _TENANT_PATH.match(path)
    assert match and match.group(1) == "demo"


@pytest.mark.parametrize("path", sorted(_NOT_TENANT_SCOPED))
def test_documented_exceptions_stay_exceptions(path):
    """If one of these ever grows a tenant segment, it needs the guard too."""
    assert "{tenant}" not in path, _NOT_TENANT_SCOPED[path]


def test_a_path_tenant_is_extracted_not_assumed():
    # The captured group is what authenticate_request compares against the
    # header, so a route with a different tenant in the URL must capture THAT
    # tenant, not the caller's.
    assert _TENANT_PATH.match("/api/acme/delay-insights").group(1) == "acme"


# ─── permission gates, through the real authenticate_request ─────────────────
#
# The middleware calls authenticate_request() for every /api request. These run
# that exact function on a real Starlette request; only the token verifier and
# the database-backed authorization are replaced, so no network or DB is used.

from fastapi import HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402

from services.configurator import security  # noqa: E402
from services.configurator.security import (  # noqa: E402
    TRAIN_PERMISSION,
    AuthContext,
    authenticate_request,
)

BATCH_REVIEW = "/api/demo/models/m3_production_delay/batch-review"
BATCH_PREDICT = "/api/demo/models/m2_inventory/batch-predict"


def _request(method: str, path: str) -> Request:
    headers = {"authorization": "Bearer test-token", "x-tenant-slug": "demo"}
    return Request({
        "type": "http", "method": method, "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "scheme": "http",
        "server": ("testserver", 80),
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
    })


def _caller(monkeypatch, *, permissions=(), roles=()):
    class _Verifier:
        def verify(self, token):
            return {"email": "caller@maxxflow.local"}

    class _Repository:
        def authorize(self, claims, tenant_slug):
            return AuthContext(
                user_id="u-1", email="caller@maxxflow.local", tenant_id="t-1",
                tenant_slug=tenant_slug, roles=frozenset(roles),
                permissions=frozenset(permissions),
            )

    monkeypatch.setattr(security, "get_jwt_verifier", lambda: _Verifier())
    monkeypatch.setattr(security, "get_auth_repository", lambda: _Repository())


@pytest.mark.parametrize("path", [BATCH_REVIEW, BATCH_REVIEW + "/"])
def test_m3_batch_review_requires_the_train_permission(path):
    assert security._required_permission(_request("POST", path)) == TRAIN_PERMISSION


def test_m3_batch_review_uses_the_same_gate_as_m2_batch_predict():
    assert (
        security._required_permission(_request("POST", BATCH_REVIEW))
        == security._required_permission(_request("POST", BATCH_PREDICT))
        == TRAIN_PERMISSION
    )


@pytest.mark.parametrize("path", [BATCH_REVIEW, BATCH_PREDICT])
def test_a_member_without_the_permission_is_refused(monkeypatch, path):
    _caller(monkeypatch)  # an active member with no roles and no permissions
    with pytest.raises(HTTPException) as refused:
        authenticate_request(_request("POST", path))
    assert refused.value.status_code == 403
    assert refused.value.detail == "Missing required permission: ml-models:train"


@pytest.mark.parametrize(
    "grant",
    [
        {"permissions": [TRAIN_PERMISSION]},
        {"permissions": ["ml-models:*"]},
        {"roles": ["maxxflow-admin"]},
    ],
    ids=["ml-models:train", "ml-models:*", "admin role"],
)
def test_a_caller_with_the_permission_is_let_through(monkeypatch, grant):
    _caller(monkeypatch, **grant)
    context = authenticate_request(_request("POST", BATCH_REVIEW))
    assert context.tenant_slug == "demo"


def test_reading_delay_insights_needs_no_permission(monkeypatch):
    _caller(monkeypatch)  # the same member refused above
    assert security._required_permission(_request("GET", "/api/demo/delay-insights")) is None
    assert authenticate_request(_request("GET", "/api/demo/delay-insights")).tenant_slug == "demo"
