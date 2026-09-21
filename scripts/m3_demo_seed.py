"""Local-only demo tooling: seed one realistic job into tenant_demo so the
real M3 pipeline (rule_engine + review agents) has something to score.

NOT synthetic-generator output (m3_production_delay has no synth.py — see
its module docstring: it reads live MRP tables directly, there is nothing
to synthesize from). This reproduces the frontend demo's "Wooden Table"
scenario: one operation logged well past its planned duration, and three
components short in the warehouse — via `_m3_demo_common.insert_job()`,
the same insert path `m3_demo_api.py`'s POST endpoint uses for
user-created MOs.

Requires a reachable DATA_DB_URL with tenant_demo already provisioned
(`maxxflow db-provision --tenant demo`, or here: a native Homebrew Postgres
already carrying that schema). Safe to re-run — it deletes any prior row
with the same JOB_REFERENCE first.

Usage:
    uv run python scripts/m3_demo_seed.py
"""

from __future__ import annotations

import sqlalchemy as sa

try:  # `python scripts/m3_demo_seed.py`-style invocation puts scripts/ on sys.path
    from _m3_demo_common import ensure_reference_data, insert_job
except ImportError:  # `python -m scripts.m3_demo_seed`-style invocation does not
    from scripts._m3_demo_common import ensure_reference_data, insert_job

TENANT = "demo"
JOB_REFERENCE = "WH/MO/00142"


def main() -> None:
    from maxxflow_data.engine import get_data_access

    da = get_data_access()
    with da.transaction(tenant=TENANT) as conn:
        # Clean up any earlier run of this script first (idempotent re-seed).
        conn.execute(sa.text(
            "DELETE FROM mo_components WHERE mo_id IN "
            "(SELECT id FROM manufacturing_orders WHERE reference = :ref)"
        ), {"ref": JOB_REFERENCE})
        conn.execute(sa.text(
            "DELETE FROM work_order_time_logs WHERE work_order_id IN "
            "(SELECT wo.id FROM work_orders wo JOIN manufacturing_orders mo ON mo.id = wo.mo_id "
            "WHERE mo.reference = :ref)"
        ), {"ref": JOB_REFERENCE})
        conn.execute(sa.text(
            "DELETE FROM work_orders WHERE mo_id IN "
            "(SELECT id FROM manufacturing_orders WHERE reference = :ref)"
        ), {"ref": JOB_REFERENCE})
        conn.execute(sa.text(
            "DELETE FROM manufacturing_orders WHERE reference = :ref"
        ), {"ref": JOB_REFERENCE})
        # Old Steel Rod/Screws/Nuts `items` rows from a prior run are left as
        # harmless orphans (insert_job() always creates fresh item rows) —
        # not worth tracking down by name just to delete on every re-seed.

        ensure_reference_data(conn)
        # expected=240min (4.0h), actual=450min (7.5h) -> the composer renders
        # "7.5 hrs logged of 4.0 planned (+3.5 hrs)", matching the frontend's
        # original mock fixture exactly.
        insert_job(
            conn,
            job_reference=JOB_REFERENCE,
            quantity=1,
            operation_name="Assemble Table",
            expected_duration_minutes=240,
            actual_duration_minutes=450,
            components=[
                {"name": "Steel Rod", "required_quantity": 10, "available_quantity": 8},
                {"name": "Screws", "required_quantity": 40, "available_quantity": 35},
                {"name": "Nuts", "required_quantity": 60, "available_quantity": 48},
            ],
        )

    print(f"seeded {JOB_REFERENCE} into tenant_{TENANT}")


if __name__ == "__main__":
    main()
