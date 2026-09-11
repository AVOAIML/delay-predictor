"""Engine + schema_router (plan §1a row 1, §12a #8).

Tenant isolation is ``SET search_path TO "<tenant_schema>", public`` — NEVER a
``WHERE tenant_id`` filter (there is no such column; the schema is per-tenant).
A read-path guard refuses any SQL that references ``tenant_id`` or the stale
``rop_status`` column (incl. ORDER BY), so those traps can't enter feature SQL.
"""

from __future__ import annotations

import functools
import re
from contextlib import contextmanager
from typing import Any, Mapping

import pandas as pd

from maxxflow_core.errors import TenantIsolationError
from maxxflow_core.settings import Settings, get_settings

_FORBIDDEN_READ = [
    (re.compile(r"\btenant_id\b", re.I), "WHERE tenant_id is forbidden — isolation is search_path (schema-per-tenant)"),
    (re.compile(r"\brop_status\b", re.I), "items.rop_status is STALE — derive ROP live, never read/ORDER BY it"),
]


def _guard_read_sql(sql: str) -> None:
    for pat, msg in _FORBIDDEN_READ:
        if pat.search(sql):
            raise TenantIsolationError(msg)


class DataAccess:
    """Implements :class:`maxxflow_core.ports.DataSource` against Postgres."""

    def __init__(self, settings: Settings | None = None, engine=None):
        self.settings = settings or get_settings()
        self._engine = engine

    @property
    def engine(self):
        if self._engine is None:
            if not self.settings.db_enabled:
                raise RuntimeError("DATA_DB_URL is empty — no database configured (DB-less mode)")
            import sqlalchemy as sa
            self._engine = sa.create_engine(
                self.settings.data_db_url,
                pool_size=self.settings.db_pool_size,
                max_overflow=self.settings.db_pool_max_overflow,
                pool_timeout=self.settings.db_pool_timeout_seconds,
                pool_recycle=self.settings.db_pool_recycle_seconds,
                pool_pre_ping=True,
            )
        return self._engine

    def _set_search_path(self, conn, tenant: str | None) -> None:
        schema = self.settings.tenant_schema(tenant)
        # Identifier is composed from a validated slug; quote defensively.
        if not re.fullmatch(r"[A-Za-z0-9_]+", schema):
            raise TenantIsolationError(f"unsafe schema name {schema!r}")
        conn.exec_driver_sql(f'SET search_path TO "{schema}", public')

    def query(self, sql: str, params: Mapping[str, Any] | None = None,
              *, tenant: str | None = None) -> pd.DataFrame:
        _guard_read_sql(sql)
        import sqlalchemy as sa
        with self.engine.connect() as conn:
            self._set_search_path(conn, tenant)
            df = pd.read_sql(sa.text(sql), conn, params=dict(params or {}))
        # psycopg returns uuid columns as uuid.UUID objects; the rest of the
        # codebase (and the synth path) treats ids as STRINGS. Normalise here so
        # DB-sourced and synthetic frames merge/compare/serialise identically.
        import uuid as _uuid
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda v: str(v) if isinstance(v, _uuid.UUID) else v)
        return df

    def execute(self, sql: str, params: Mapping[str, Any] | None = None,
                *, tenant: str | None = None) -> None:
        import sqlalchemy as sa
        with self.engine.begin() as conn:
            self._set_search_path(conn, tenant)
            conn.execute(sa.text(sql), dict(params or {}))

    @contextmanager
    def transaction(self, *, tenant: str | None = None):
        """One connection, one outer transaction, for a batch of writes that
        would otherwise pay execute()'s per-call connection+BEGIN+COMMIT cost
        once per row. Yields the raw connection so the caller can isolate each
        statement with its own ``conn.begin_nested()`` (SAVEPOINT) — a failure
        there rolls back only that nested block, not the whole batch."""
        with self.engine.begin() as conn:
            self._set_search_path(conn, tenant)
            yield conn


@functools.lru_cache(maxsize=1)
def get_data_access() -> DataAccess:
    """One DataAccess—and therefore one bounded connection pool—per process."""
    return DataAccess()
