"""Local-only demo tooling: seed two realistic jobs into tenant_demo so the
real M3 pipeline (rule engine + review evidence) has something to score.

NOT synthetic-generator output (m3_production_delay has no synth.py — see
its module docstring: it reads live MRP tables directly, there is nothing
to synthesize from). This reproduces the frontend demo's "Wooden Table"
scenario: one operation logged well past its planned duration, and three
components short in the warehouse — via `_m3_demo_common.insert_job()`,
the same insert path `m3_demo_api.py`'s POST endpoint uses for
user-created MOs. The second job reproduces the user-story cascading-delay
case: an independent predecessor is 50% complete and running at 1.30x its
progress-adjusted plan, while its dependent operation has not started. With
the dependent's planned duration included, combined duration-weighted MO
progress is 28.57%, clearing the 25% analysis gate.

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
    from _m3_demo_common import ensure_reference_data, insert_cascading_delay_job, insert_job
except ImportError:  # `python -m scripts.m3_demo_seed`-style invocation does not
    from scripts._m3_demo_common import ensure_reference_data, insert_cascading_delay_job, insert_job

TENANT = "demo"
JOB_REFERENCE = "WH/MO/00142"
CASCADE_JOB_REFERENCE = "WH/MO/CASCADE-001"


def _verify_cascade_seed() -> None:
    """Print the calculated cascade facts without making an LLM call."""
    from m3_production_delay.review.composer import compose
    from m3_production_delay.review.evidence import build_evidence
    from m3_production_delay.review.schemas import SIGNAL_CRITICAL_PATH_CASCADE
    from m3_production_delay.review.validators import validate
    from m3_production_delay.rule_engine.dal import read_delay_tables
    from m3_production_delay.rule_engine.elements import calculate_delay_elements_for_jobs
    from m3_production_delay.rule_engine.rollup import build_job_rollups

    tables, md = read_delay_tables(TENANT)
    rollups = build_job_rollups(tables, md, job_references=[CASCADE_JOB_REFERENCE])
    if len(rollups) != 1:
        raise RuntimeError(
            f"expected one rollup for {CASCADE_JOB_REFERENCE}, got {len(rollups)}"
        )
    scored = calculate_delay_elements_for_jobs(rollups)
    job = scored[0]
    predecessor = next(
        op for op in job["operations"] if op["operation_type"] == "INDEPENDENT"
    )
    dependent = next(
        op for op in job["operations"] if op["operation_type"] == "DEPENDENT"
    )
    pack = build_evidence(job, {
        "time_overrun_ratio": 0.50,
        "operator_pace_ratio": 0.35,
        "material_shortfall_ratio": 0.10,
        "supplier_reliability": 0.05,
    }, 1.0)
    draft = compose(pack)
    validation_issues = validate(pack, draft)
    cascade_lines = [
        line for line in draft.why_lines
        if line.signal_key == SIGNAL_CRITICAL_PATH_CASCADE
    ]

    print("cascade verification:")
    print(f"  manufacturing_order_progress={job['manufacturing_order_progress']:.2%}")
    print(
        "  predecessor "
        f"work_done={predecessor['current_done_quantity'] / predecessor['job_quantity']:.0%} "
        f"time_overrun_ratio={predecessor['time_overrun_ratio']:.2f} "
        f"risk_score={predecessor['composite_risk_score']:.4f} "
        f"is_delayed={predecessor['is_delayed']}"
    )
    print(
        "  dependent "
        f"status={dependent['status']} "
        f"own_time_overrun={dependent['time_overrun_ratio']} "
        f"predecessor_time_overrun={dependent['predecessor_time_overrun_ratio']:.2f} "
        f"critical_path={dependent['is_on_critical_path']} "
        f"cascade_ratio={dependent['critical_path_cascade_ratio']:.2f} "
        f"base_risk_score={dependent['base_composite_risk_score']:.4f} "
        f"risk_score={dependent['composite_risk_score']:.4f} "
        f"is_delayed={dependent['is_delayed']}"
    )
    print(
        "  review "
        f"visible_cascade_reason={len(cascade_lines) == 1} "
        f"message={cascade_lines[0].text!r}"
    )
    print(
        "  validation "
        f"issues={[(issue.check, issue.severity, issue.message) for issue in validation_issues]}"
    )


def main() -> None:
    from maxxflow_data.engine import get_data_access

    da = get_data_access()
    with da.transaction(tenant=TENANT) as conn:
        # Clean up any earlier run of this script first (idempotent re-seed).
        refs = [
            JOB_REFERENCE,
            f"{JOB_REFERENCE}-H1",
            f"{JOB_REFERENCE}-H2",
            f"{JOB_REFERENCE}-H3",
            CASCADE_JOB_REFERENCE,
            f"{CASCADE_JOB_REFERENCE}-H1",
            f"{CASCADE_JOB_REFERENCE}-H2",
            f"{CASCADE_JOB_REFERENCE}-H3",
        ]
        supplier_reference_pattern = f"DEMO-%-{JOB_REFERENCE[-5:]}-%"
        conn.execute(sa.text(
            "DELETE FROM goods_received_notes WHERE reference_no LIKE :pattern"
        ), {"pattern": supplier_reference_pattern})
        conn.execute(sa.text(
            "DELETE FROM purchase_order_lines WHERE purchase_order_id IN "
            "(SELECT id FROM purchase_orders WHERE reference_no LIKE :pattern)"
        ), {"pattern": supplier_reference_pattern})
        conn.execute(sa.text(
            "DELETE FROM purchase_orders WHERE reference_no LIKE :pattern"
        ), {"pattern": supplier_reference_pattern})
        conn.execute(sa.text(
            "DELETE FROM item_vendors WHERE item_id IN "
            "(SELECT mc.item_id FROM mo_components mc "
            "JOIN manufacturing_orders mo ON mo.id = mc.mo_id WHERE mo.reference = :ref)"
        ), {"ref": JOB_REFERENCE})
        conn.execute(sa.text(
            "DELETE FROM mo_components WHERE mo_id IN "
            "(SELECT id FROM manufacturing_orders WHERE reference = ANY(:refs))"
        ), {"refs": refs})
        conn.execute(sa.text(
            "DELETE FROM work_order_time_logs WHERE work_order_id IN "
            "(SELECT wo.id FROM work_orders wo JOIN manufacturing_orders mo ON mo.id = wo.mo_id "
            "WHERE mo.reference = ANY(:refs))"
        ), {"refs": refs})
        conn.execute(sa.text(
            "DELETE FROM operation_dependencies WHERE operation_id IN "
            "(SELECT wo.operation_id FROM work_orders wo "
            "JOIN manufacturing_orders mo ON mo.id = wo.mo_id WHERE mo.reference = ANY(:refs)) "
            "OR depends_on_id IN "
            "(SELECT wo.operation_id FROM work_orders wo "
            "JOIN manufacturing_orders mo ON mo.id = wo.mo_id WHERE mo.reference = ANY(:refs))"
        ), {"refs": refs})
        conn.execute(sa.text(
            "DELETE FROM work_orders WHERE mo_id IN "
            "(SELECT id FROM manufacturing_orders WHERE reference = ANY(:refs))"
        ), {"refs": refs})
        conn.execute(sa.text(
            "DELETE FROM manufacturing_orders WHERE reference = ANY(:refs)"
        ), {"refs": refs})
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
            product_name="Wooden Table",
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
        insert_cascading_delay_job(conn, job_reference=CASCADE_JOB_REFERENCE)

    print(f"seeded {JOB_REFERENCE} and {CASCADE_JOB_REFERENCE} into tenant_{TENANT}")
    _verify_cascade_seed()


if __name__ == "__main__":
    main()
