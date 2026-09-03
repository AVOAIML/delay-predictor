"""maxxflow_data — the DAL: the ONE place that reads/writes tenant Postgres.

Enforces the plan's §1a / §12a invariants at the boundary so no module can get
them wrong: tenant isolation by ``SET search_path`` (never ``WHERE tenant_id``),
the single as-of clock, ``decimal.Decimal`` math, ROP derived live (the stale
``rop_status`` column is forbidden in reads), operator UUIDs HMAC→tier *before*
bronze, GRN on-time keyed off the status transition, ``len(allowedEmployees)``
with a 0-guard, and ``deleted_at`` exclusion.
"""

from maxxflow_data.engine import DataAccess, get_data_access

__all__ = ["DataAccess", "get_data_access"]
