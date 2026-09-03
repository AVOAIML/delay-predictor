"""Emit Postgres DDL + pandas/pandera type maps from ``schema_def`` (one source)."""

from __future__ import annotations

import re

from maxxflow_data.schema_def import Col, PUBLIC_TABLES, TENANT_TABLES


def _pg_type(t: str) -> str:
    if t == "uuid":
        return "uuid"
    if t == "text":
        return "text"
    if t == "int":
        return "integer"
    if t == "bool":
        return "boolean"
    if t == "jsonb":
        return "jsonb"
    if t == "float":
        return "double precision"
    if t == "date":
        return "date"
    if t == "timestamp":
        return "timestamp(6)"
    if t == "text[]":
        return "text[]"
    if t == "uuid[]":
        return "uuid[]"
    if t.startswith("varchar"):
        return t
    if t.startswith("decimal"):
        return "numeric" + t[len("decimal"):]
    raise ValueError(f"unknown logical type {t!r}")


def _col_ddl(c: Col) -> str:
    parts = [f'"{c.name}"', _pg_type(c.type)]
    if not c.nullable:
        parts.append("NOT NULL")
    if c.default is not None:
        parts.append(f"DEFAULT {c.default}")
    return " ".join(parts)


def _table_ddl(schema: str, table: str, cols: list[Col]) -> str:
    body = ",\n  ".join(_col_ddl(c) for c in cols)
    return (
        f'CREATE TABLE IF NOT EXISTS "{schema}"."{table}" (\n  {body},\n'
        f'  PRIMARY KEY ("id")\n);'
    )


def tenant_template_ddl(schema: str) -> str:
    out = [f'CREATE SCHEMA IF NOT EXISTS "{schema}";']
    for table, cols in TENANT_TABLES.items():
        out.append(_table_ddl(schema, table, cols))
    return "\n\n".join(out)


def public_ddl(schema: str = "public") -> str:
    out = [f'CREATE SCHEMA IF NOT EXISTS "{schema}";']
    for table, cols in PUBLIC_TABLES.items():
        out.append(_table_ddl(schema, table, cols))
    return "\n\n".join(out)


# --- pandas dtype map for gate A / contract checks --------------------------
def pandas_dtype(t: str) -> str:
    if t in ("int",):
        return "Int64"
    if t == "float" or t.startswith("decimal"):
        return "float64"
    if t == "bool":
        return "boolean"
    if t in ("timestamp", "date"):
        return "datetime64[ns]"
    return "object"


def decimal_scale(t: str) -> int | None:
    m = re.match(r"decimal\(\d+,(\d+)\)", t)
    return int(m.group(1)) if m else None
