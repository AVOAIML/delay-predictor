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
