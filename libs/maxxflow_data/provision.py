"""Provision public + tenant schema locally (== the prod per-tenant migration).

The real app applies the tenant_template via Prisma migrate to ``tenant_<slug>``.
Locally we emit the same DDL from ``schema_def`` and run it after ``SET search_path``
— so feature SQL is identical to production.
"""

from __future__ import annotations

from maxxflow_data.ddl import public_ddl, tenant_template_ddl
from maxxflow_data.engine import get_data_access


def _run_script(conn, sql: str) -> None:
    for stmt in (s.strip() for s in sql.split(";")):
        if stmt:
            conn.exec_driver_sql(stmt)


def provision_tenant(tenant: str | None = None) -> str:
    da = get_data_access()
    schema = da.settings.tenant_schema(tenant)
    with da.engine.begin() as conn:
        _run_script(conn, public_ddl())
        _run_script(conn, tenant_template_ddl(schema))
    return schema
