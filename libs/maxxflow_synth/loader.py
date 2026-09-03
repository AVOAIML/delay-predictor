"""Seed synthetic ground-truth INTO the tenant Postgres schema (plan §5).

Same path production reads — the pipeline cannot tell synthetic from real. In
DB-less mode (no DATA_DB_URL, e.g. CI sandbox) it simulates and returns the
row-count summary without writing, so gates/tests still run.

NOTE: the actual Postgres write is exercised on the docker stack (`make seed`),
not in the CI sandbox (which has no Postgres). See NOTES.md.
"""

from __future__ import annotations

from maxxflow_core.settings import get_settings
from maxxflow_data.schema_def import PUBLIC_TABLES, TENANT_TABLES
from maxxflow_synth.simulators import simulate

_MODULES = ["m1_quote", "m2_inventory", "m3_delay", "m4_bom"]


def _sqla_dtypes(table_cols):
    """SQLAlchemy type hints for to_sql. Needed for uuid / array / jsonb, AND for
    timestamp/date columns — an all-NULL datetime column would otherwise be typed
    as text by pandas and sent as VARCHAR into a timestamp column."""
    from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
    from sqlalchemy.types import Date, DateTime, Text
    out = {}
    for c in table_cols:
        if c.type == "uuid":
            out[c.name] = UUID(as_uuid=False)
        elif c.type == "uuid[]":
            out[c.name] = ARRAY(UUID(as_uuid=False))
        elif c.type == "text[]":
            out[c.name] = ARRAY(Text())
        elif c.type == "jsonb":
            out[c.name] = JSONB()
        elif c.type == "timestamp":
            out[c.name] = DateTime()
        elif c.type == "date":
            out[c.name] = Date()
    return out


def _fill_required(df, engine, schema: str, table: str, spec_cols):
    """Supply bookkeeping columns the destination requires but does not default.

    The local provisioner (schema_def -> ddl.py) gives ``id`` DEFAULT
    gen_random_uuid(), ``updated_at`` DEFAULT now() and ``created_by`` a zero-UUID
    default, so no simulator emits them. Prisma does NOT: ``@default(uuid())``,
    ``@updatedAt`` and app-set ``created_by`` are APPLICATION-level, so the
    production DDL carries no DB default for any of them. Seeding into a
    Prisma-generated schema therefore dies on the first INSERT with a
    NotNullViolation on ``updated_at`` — a column nobody wrote, in a table nobody
    changed. It is not a synth bug and not a schema bug; it is the seam between
    the two, and it only appears against the real schema.

    Rather than teach twenty simulators about bookkeeping, ask the destination what
    it requires and fill from the intent schema_def already records in Col.default.
    Against the local DDL this matches nothing and costs one catalog query.

    A required column with NO declared default is not invented — guessing at a
    business column would put fabricated values in a table that reads as real.
    """
    import uuid
    from datetime import datetime, timezone

    import sqlalchemy as sa

    with engine.connect() as conn:
        required = list(conn.execute(sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = :s AND table_name = :t "
            "  AND is_nullable = 'NO' AND column_default IS NULL"
        ), {"s": schema, "t": table}).scalars())

    missing = [c for c in required if c not in df.columns]
    if not missing:
        return df

    spec = {c.name: c for c in spec_cols}
    df = df.copy()
    undeclared = []
    for name in missing:
        col = spec.get(name)
        raw = col.default if col is not None else None
        if raw is None:
            undeclared.append(name)
            continue
        lowered = raw.strip().lower()
        if lowered in ("now()", "current_timestamp"):
            df[name] = datetime.now(timezone.utc).replace(tzinfo=None)
        elif lowered == "gen_random_uuid()":
            df[name] = [str(uuid.uuid4()) for _ in range(len(df))]
        else:
            df[name] = raw.strip().strip("'")

    if undeclared:
        raise ValueError(
            f'{schema}.{table}: the destination requires {", ".join(sorted(undeclared))}, '
            "which no simulator produces and schema_def declares no default for. Add it "
            "to the simulator if it carries meaning, or give it a default in schema_def "
            "if it is bookkeeping. Refusing to invent a value."
        )
    return df


def _new_rows_only(df, engine, schema: str, table: str):
    """Drop rows whose id already exists — makes seeding idempotent. MasterData is
    shared across modules (same deterministic ids) and re-runs shouldn't collide."""
    if df is None or df.empty or "id" not in df.columns:
        return df
    import pandas as pd
    import sqlalchemy as sa
    with engine.connect() as conn:
        existing = pd.read_sql(sa.text(f'SELECT id FROM "{schema}"."{table}"'), conn)
    if existing.empty:
        return df
    have = set(existing["id"].astype(str))
    return df[~df["id"].astype(str).isin(have)]


def _load_tables(da, tenant: str, batch) -> dict:
    counts = {}
    schema = da.settings.tenant_schema(tenant)
    # public tables first (cross-schema refs), then tenant tables.
    for name, df in batch.public_tables.items():
        df = _new_rows_only(df, da.engine, "public", name)
        if not df.empty:
            df = _fill_required(df, da.engine, "public", name, PUBLIC_TABLES[name])
            df.to_sql(name, da.engine, schema="public", if_exists="append", index=False,
                      dtype=_sqla_dtypes(PUBLIC_TABLES[name]), method="multi", chunksize=500)
        counts[f"public.{name}"] = len(df)
    for name, df in batch.tables.items():
        if name.startswith("_"):   # _meta / _labels / _milestone are metadata, not tables
            continue
        df = _new_rows_only(df, da.engine, schema, name)
        if not df.empty:
            df = _fill_required(df, da.engine, schema, name, TENANT_TABLES.get(name, []))
            df.to_sql(name, da.engine, schema=schema, if_exists="append", index=False,
                      dtype=_sqla_dtypes(TENANT_TABLES.get(name, [])), method="multi", chunksize=500)
        counts[name] = len(df)
    return counts


def seed(tenant: str = "demo", module: str = "all", seed: int = 7) -> dict:
    modules = _MODULES if module == "all" else [module]
    settings = get_settings()
    da = None
    if settings.db_enabled:
        from maxxflow_data.engine import get_data_access
        da = get_data_access()

    summary = {"tenant": tenant, "db_write": bool(da), "modules": {}}
    for m in modules:
        batch = simulate(m, seed=seed, tenant=tenant)
        info = {"label_kind": batch.label_kind, "n_rows": int(len(batch.label)),
                "positive_rate": round(float(batch.label.mean()), 4),
                "has_raw_tables": bool(batch.tables)}
        if da and batch.tables:
            info["loaded"] = _load_tables(da, tenant, batch)
        elif da and not batch.tables:
            info["note"] = batch.meta.get("note", "raw tables not yet implemented")
        summary["modules"][m] = info
    return summary
