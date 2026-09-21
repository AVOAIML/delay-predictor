"""Seeds realistic M3 production-delay rows into a tenant schema.

``maxxflow seed --module m3_delay`` does NOT cover this: ``simulate_m3`` emits a
flat feature/label frame for the classical-ML path and no raw tables at all
(its own ``meta`` says so). The rule engine reads work orders, time logs,
components, vendors and purchase orders, so a local end-to-end run of M3 needs
those rows to exist.

Four jobs, chosen so one run exercises every status the review pipeline can
produce:

  WH/MO/00142  Cutting is overrunning with a slow operator, steel is short and
               its vendor delivers late; welding is on track; assembly has not
               started and depends on cutting.
               -> lines for time, operator, material and supplier, plus a
                  not-started-but-delayed warning and a cascade note.

  WH/MO/00143  Everything on schedule, every component in stock.
               -> no lines at all: nothing fired, nothing to explain.

  WH/MO/00144  Every operation still PENDING, nothing logged.
               -> suppressed_not_scorable (the 25% gate), with the engine's
                  numbers still carried for the panel.

  WH/MO/00145  A component with ZERO stock, which makes the shortfall ratio
               infinite and the composite score with it.
               -> the infinity is coerced to null and flagged, and the
                  shortage still reaches the material list.

Idempotent: every row uses a deterministic uuid5, so re-running replaces the
same rows rather than accumulating new ones. Run it as often as you like.

    uv run python scripts/seed_m3_demo.py --tenant demo
    uv run python scripts/seed_m3_demo.py --tenant demo --clean   # remove only
"""

from __future__ import annotations

import argparse
import datetime as dt
import uuid

import sqlalchemy as sa

from maxxflow_core.masterdata import CATEGORIES
from maxxflow_synth.masterdata_seed import build_masterdata

_NS = uuid.UUID("2f8c1a5e-7b3d-4c9a-8e1f-6d2b0a4c7e91")  # this script's own namespace

JOB_REFERENCES = ["WH/MO/00142", "WH/MO/00143", "WH/MO/00144", "WH/MO/00145"]


def _u(*parts: str) -> str:
    """Deterministic id, so re-seeding overwrites instead of duplicating."""
    return str(uuid.uuid5(_NS, ":".join(parts)))


def _days_ago(n: float) -> dt.datetime:
    return dt.datetime.now() - dt.timedelta(days=n)


# --- master data -------------------------------------------------------------


def _seed_master_data(conn, tenant: str) -> dict:
    """Insert the platform's own MasterData vocabulary, with the same uuid5 ids
    ``maxxflow_synth`` uses, so a seed from either source resolves identically
    through ``load_md_map``."""
    categories, values, mapping = build_masterdata(tenant)
    for _, row in categories.iterrows():
        conn.execute(
            sa.text(
                "INSERT INTO master_data_category (id, name, code, status, is_system_fixed) "
                "VALUES (:id, :name, :code, 'active', true) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": row["id"], "name": row["name"], "code": row["code"]},
        )
    for _, row in values.iterrows():
        conn.execute(
            sa.text(
                "INSERT INTO master_data (id, category_id, name, code, status, is_system_fixed) "
                "VALUES (:id, :category_id, :name, :code, 'active', true) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": row["id"],
                "category_id": row["category_id"],
                "name": row["name"],
                "code": row["code"],
            },
        )
    return {key: value for key, value in mapping.code_to_id.items()}


# --- reference rows shared by every job ---------------------------------------


def _seed_shared(conn, tenant: str, md: dict) -> dict:
    """Work centre, items, vendors and purchase history. Two vendors with
    deliberately different records: one chronically late, one early."""
    ids = {
        "work_center": _u(tenant, "wc", "cutting-bay"),
        "steel": _u(tenant, "item", "steel-plate"),
        "bolt": _u(tenant, "item", "bolt-m8"),
        "resin": _u(tenant, "item", "epoxy-resin"),
        "vendor_late": _u(tenant, "vendor", "lanka-steel"),
        "vendor_ontime": _u(tenant, "vendor", "colombo-fasteners"),
        "warehouse": _u(tenant, "warehouse", "main"),
        "product": _u(tenant, "product", "frame"),
        "bom": _u(tenant, "bom", "frame"),
        "working_hours": md[("WORKING_HOURS", "STANDARD_8H")]
        if ("WORKING_HOURS", "STANDARD_8H") in md
        else _u(tenant, "working-hours", "default"),
    }

    conn.execute(
        sa.text(
            "INSERT INTO work_centers (id, name, code, working_hours_id, setup_time, cleanup_time) "
            "VALUES (:id, 'Cutting Bay', 'WC-CUT-01', :wh, 15, 10) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ids["work_center"], "wh": ids["working_hours"]},
    )

    items = [
        (ids["steel"], "Steel Plate 12mm", "STL-12MM", 40),   # short against 100 required
        (ids["bolt"], "Bolt M8", "BLT-M8", 5000),             # plenty
        (ids["resin"], "Epoxy Resin", "RES-EPX", 0),          # ZERO stock -> infinite ratio
    ]
    for item_id, name, part_number, quantity in items:
        conn.execute(
            sa.text(
                "INSERT INTO items (id, item_name, part_number, available_quantity) "
                "VALUES (:id, :name, :pn, :qty) "
                "ON CONFLICT (id) DO UPDATE SET available_quantity = EXCLUDED.available_quantity"
            ),
            {"id": item_id, "name": name, "pn": part_number, "qty": quantity},
        )

    vendors = [
        (_u(tenant, "iv", "steel"), ids["steel"], ids["vendor_late"], "Lanka Steel", 14),
        (_u(tenant, "iv", "bolt"), ids["bolt"], ids["vendor_ontime"], "Colombo Fasteners", 7),
        (_u(tenant, "iv", "resin"), ids["resin"], ids["vendor_late"], "Lanka Steel", 21),
    ]
    for link_id, item_id, vendor_id, vendor_name, lead_time in vendors:
        conn.execute(
            sa.text(
                "INSERT INTO item_vendors (id, item_id, vendor_id, vendor_name, lead_time_days, "
                "unit_price) VALUES (:id, :item, :vendor, :name, :lead, 12.50) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": link_id,
                "item": item_id,
                "vendor": vendor_id,
                "name": vendor_name,
                "lead": lead_time,
            },
        )

    # Purchase history — what vendor_lead_time_ratio is computed from.
    # Lanka Steel: promised 14 days, took 21 -> 1.5. Colombo: promised 14,
    # took 11 -> 0.79.
    purchase_orders = [
        ("PO-8801", ids["vendor_late"], ids["steel"], 40, 14, 21),
        ("PO-8802", ids["vendor_ontime"], ids["bolt"], 40, 14, 11),
        ("PO-8803", ids["vendor_late"], ids["resin"], 30, 14, 22),
    ]
    for reference, vendor_id, item_id, ordered_days_ago, promised, actual in purchase_orders:
        po_id = _u(tenant, "po", reference)
        ordered_at = _days_ago(ordered_days_ago)
        conn.execute(
            sa.text(
                "INSERT INTO purchase_orders (id, reference_no, vendor_id, warehouse_id, "
                "scheduled_delivery_date, status, sent_at, created_at) "
                "VALUES (:id, :ref, :vendor, :wh, :deadline, 'Partially Received', :sent, :sent) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": po_id,
                "ref": reference,
                "vendor": vendor_id,
                "wh": ids["warehouse"],
                "deadline": ordered_at + dt.timedelta(days=promised),
                "sent": ordered_at,
            },
        )
        conn.execute(
            sa.text(
                "INSERT INTO purchase_order_lines (id, purchase_order_id, item_id, item_name, "
                "ordered_quantity, received_quantity, unit_price) "
                "VALUES (:id, :po, :item, 'seeded line', 100, 60, 12.50) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"id": _u(tenant, "pol", reference), "po": po_id, "item": item_id},
        )
        # grn_on_time keys off the status transition timestamp (updated_at),
        # gated on a received status — never a "received date" column, which
        # does not exist anywhere in the schema.
        conn.execute(
            sa.text(
                "INSERT INTO goods_received_notes (id, reference_no, purchase_order_id, vendor_id, "
                "warehouse_id, scheduled_delivery_date, status, is_partially_closed, created_at, "
                "updated_at) VALUES (:id, :ref, :po, :vendor, :wh, :deadline, 'Goods Received', "
                "false, :created, :received) ON CONFLICT (id) DO UPDATE SET updated_at = "
                "EXCLUDED.updated_at"
            ),
            {
                "id": _u(tenant, "grn", reference),
                "ref": reference.replace("PO", "GRN"),
                "po": po_id,
                "vendor": vendor_id,
                "wh": ids["warehouse"],
                "deadline": ordered_at + dt.timedelta(days=promised),
                "created": ordered_at,
                "received": ordered_at + dt.timedelta(days=actual),
            },
        )
    return ids


# --- jobs ---------------------------------------------------------------------


def _seed_operator_history(conn, tenant: str, shared: dict, md: dict) -> dict:
    """Two operators with a track record: one consistently over schedule
    (pace 1.25), one slightly under (0.95). operator_pace_ratio is the mean of
    elapsed/scheduled over their COMPLETED work orders, so these need a
    finished job of their own to be computed from."""
    operators = {"slow": _u(tenant, "operator", "slow"), "steady": _u(tenant, "operator", "steady")}
    history_mo = _u(tenant, "mo", "HISTORY")
    conn.execute(
        sa.text(
            "INSERT INTO manufacturing_orders (id, reference, product_id, bom_id, quantity, "
            "status_id, component_status_id) VALUES (:id, 'WH/MO/00100', :product, :bom, 50, "
            ":status, :availability) ON CONFLICT (id) DO NOTHING"
        ),
        {
            "id": history_mo,
            "product": shared["product"],
            "bom": shared["bom"],
            "status": md[("MO_STATUS", "DONE")],
            "availability": md[("MO_COMPONENT_STATUS", "AVAILABLE")],
        },
    )
    history_op = _u(tenant, "operation", "history")
    conn.execute(
        sa.text(
            'INSERT INTO operations (id, bom_id, "operationName", "workCenterId", '
            '"estimatedDuration", operation_type_id, allowed_employees) '
            "VALUES (:id, :bom, 'Historic Cutting', :wc, 400, :type, ARRAY[]::uuid[]) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {
            "id": history_op,
            "bom": shared["bom"],
            "wc": shared["work_center"],
            "type": md[("OPERATION_TYPE", "INDEPENDENT")],
        },
    )
    for label, elapsed in (("slow", 500), ("steady", 380)):
        wo_id = _u(tenant, "wo", f"history-{label}")
        conn.execute(
            sa.text(
                "INSERT INTO work_orders (id, mo_id, operation_id, work_center_id, quantity, "
                "units_done, expected_duration, real_duration, actual_start, actual_end, "
                "assigned_operators, status_id) VALUES (:id, :mo, :op, :wc, 50, 50, 400, :real, "
                ":start, :end, ARRAY[:operator]::text[], :status) "
                "ON CONFLICT (id) DO UPDATE SET real_duration = EXCLUDED.real_duration"
            ),
            {
                "id": wo_id,
                "mo": history_mo,
                "op": history_op,
                "wc": shared["work_center"],
                "real": elapsed,
                "start": _days_ago(30),
                "end": _days_ago(29),
                "operator": operators[label],
                "status": md[("WORK_ORDER_STATUS", "DONE")],
            },
        )
        conn.execute(
            sa.text(
                "INSERT INTO work_order_time_logs (id, work_order_id, operator_id, started_at, "
                "ended_at, duration_minutes) VALUES (:id, :wo, :operator, :start, :end, :minutes) "
                "ON CONFLICT (id) DO UPDATE SET duration_minutes = EXCLUDED.duration_minutes"
            ),
            {
                "id": _u(tenant, "tl", f"history-{label}"),
                "wo": wo_id,
                "operator": operators[label],
                "start": _days_ago(30),
                "end": _days_ago(29),
                "minutes": elapsed,
            },
        )
    return operators


def _seed_job(
    conn,
    tenant: str,
    shared: dict,
    md: dict,
    operators: dict,
    *,
    reference: str,
    components: list[tuple[str, float]],
    operations: list[dict],
) -> None:
    """One manufacturing order with its components and work orders.

    ``components`` is MO-level, exactly as the real schema stores it — which is
    why the rollup hands the same list to every operation and Section 3
    de-duplicates it back to job scope.
    """
    mo_id = _u(tenant, "mo", reference)
    conn.execute(
        sa.text(
            "INSERT INTO manufacturing_orders (id, reference, product_id, bom_id, quantity, "
            "scheduled_date, status_id, component_status_id) VALUES (:id, :ref, :product, :bom, "
            "100, :scheduled, :status, :availability) ON CONFLICT (id) DO NOTHING"
        ),
        {
            "id": mo_id,
            "ref": reference,
            "product": shared["product"],
            "bom": shared["bom"],
            "scheduled": _days_ago(-5),
            "status": md[("MO_STATUS", "IN_PROGRESS")],
            "availability": md[("MO_COMPONENT_STATUS", "PARTIALLY_AVAILABLE")],
        },
    )

    for item_key, required in components:
        conn.execute(
            sa.text(
                "INSERT INTO mo_components (id, mo_id, item_id, component_type, required_qty, "
                "reserved_qty, availability_id) VALUES (:id, :mo, :item, 'item', :required, 0, "
                ":availability) ON CONFLICT (id) DO UPDATE SET required_qty = EXCLUDED.required_qty"
            ),
            {
                "id": _u(tenant, "moc", reference, item_key),
                "mo": mo_id,
                "item": shared[item_key],
                "required": required,
                "availability": md[("MO_COMPONENT_STATUS", "PARTIALLY_AVAILABLE")],
            },
        )

    operation_ids: dict[str, str] = {}
    for spec in operations:
        operation_id = _u(tenant, "operation", reference, spec["name"])
        operation_ids[spec["name"]] = operation_id
        conn.execute(
            sa.text(
                'INSERT INTO operations (id, bom_id, "operationName", "workCenterId", '
                '"estimatedDuration", operation_type_id, allowed_employees) '
                "VALUES (:id, :bom, :name, :wc, :expected, :type, ARRAY[]::uuid[]) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": operation_id,
                "bom": shared["bom"],
                "name": spec["name"],
                "wc": shared["work_center"],
                "expected": spec["expected"],
                "type": md[
                    ("OPERATION_TYPE", "DEPENDENT" if spec.get("depends_on") else "INDEPENDENT")
                ],
            },
        )

    # operation_dependencies is defined at the BOM level, so it is written once
    # the operation rows exist and resolved back to THIS job's work orders by
    # rollup.py.
    for spec in operations:
        if not spec.get("depends_on"):
            continue
        conn.execute(
            sa.text(
                "INSERT INTO operation_dependencies (id, operation_id, depends_on_id) "
                "VALUES (:id, :op, :depends_on) ON CONFLICT (id) DO NOTHING"
            ),
            {
                "id": _u(tenant, "dep", reference, spec["name"]),
                "op": operation_ids[spec["name"]],
                "depends_on": operation_ids[spec["depends_on"]],
            },
        )

    for spec in operations:
        wo_id = _u(tenant, "wo", reference, spec["name"])
        operator = operators[spec["operator"]] if spec.get("operator") else None
        conn.execute(
            sa.text(
                "INSERT INTO work_orders (id, mo_id, operation_id, work_center_id, quantity, "
                "units_done, expected_duration, real_duration, actual_start, assigned_operators, "
                "status_id) VALUES (:id, :mo, :op, :wc, 100, :done, :expected, :real, :start, "
                ":operators, :status) ON CONFLICT (id) DO UPDATE SET "
                "real_duration = EXCLUDED.real_duration, units_done = EXCLUDED.units_done, "
                "status_id = EXCLUDED.status_id"
            ),
            {
                "id": wo_id,
                "mo": mo_id,
                "op": operation_ids[spec["name"]],
                "wc": shared["work_center"],
                "done": spec["units_done"],
                "expected": spec["expected"],
                "real": spec["actual"],
                "start": _days_ago(2) if spec["actual"] else None,
                "operators": [operator] if operator else [],
                "status": md[("WORK_ORDER_STATUS", spec["status"])],
            },
        )
        if spec["actual"] and operator:
            conn.execute(
                sa.text(
                    "INSERT INTO work_order_time_logs (id, work_order_id, operator_id, "
                    "started_at, ended_at, duration_minutes) "
                    "VALUES (:id, :wo, :operator, :start, :end, :minutes) "
                    "ON CONFLICT (id) DO UPDATE SET duration_minutes = EXCLUDED.duration_minutes"
                ),
                {
                    "id": _u(tenant, "tl", reference, spec["name"]),
                    "wo": wo_id,
                    "operator": operator,
                    "start": _days_ago(2),
                    "end": _days_ago(1),
                    "minutes": spec["actual"],
                },
            )


JOBS = [
    {
        # Cutting overrunning (600 logged of 480 planned = 1.25) with the slow
        # operator; welding 30% elapsed and on track; assembly not started and
        # dependent on cutting. Steel is 60 short and its vendor runs late.
        "reference": "WH/MO/00142",
        "components": [("steel", 100.0), ("bolt", 500.0)],
        "operations": [
            {"name": "Cutting", "expected": 480, "actual": 600, "units_done": 60,
             "status": "IN_PROGRESS", "operator": "slow"},
            {"name": "Welding", "expected": 300, "actual": 90, "units_done": 35,
             "status": "IN_PROGRESS", "operator": "steady"},
            {"name": "Assembly", "expected": 240, "actual": None, "units_done": 0,
             "status": "PENDING", "operator": "slow", "depends_on": "Cutting"},
        ],
    },
    {
        # Nothing fires: on schedule, steady operator, everything in stock.
        "reference": "WH/MO/00143",
        "components": [("bolt", 200.0)],
        "operations": [
            {"name": "Cutting", "expected": 480, "actual": 300, "units_done": 70,
             "status": "IN_PROGRESS", "operator": "steady"},
        ],
    },
    {
        # Nothing logged anywhere -> below the 25% scoring gate on every
        # operation -> suppressed_not_scorable.
        "reference": "WH/MO/00144",
        "components": [("steel", 80.0)],
        "operations": [
            {"name": "Cutting", "expected": 480, "actual": None, "units_done": 0,
             "status": "PENDING", "operator": "slow"},
            {"name": "Welding", "expected": 300, "actual": None, "units_done": 0,
             "status": "PENDING", "operator": "slow", "depends_on": "Cutting"},
        ],
    },
    {
        # Epoxy resin has ZERO stock, so material_shortfall_ratio is infinite
        # and the composite score with it.
        "reference": "WH/MO/00145",
        "components": [("resin", 150.0)],
        "operations": [
            {"name": "Moulding", "expected": 360, "actual": 450, "units_done": 40,
             "status": "IN_PROGRESS", "operator": "slow"},
        ],
    },
]


def _clean(conn, tenant: str) -> None:
    """Remove exactly the rows this script writes, child tables first."""
    references = JOB_REFERENCES + ["WH/MO/00100"]
    mo_ids = [_u(tenant, "mo", reference) for reference in references]
    mo_ids.append(_u(tenant, "mo", "HISTORY"))
    conn.execute(
        sa.text(
            "DELETE FROM work_order_time_logs WHERE work_order_id IN "
            "(SELECT id FROM work_orders WHERE mo_id = ANY(:mos))"
        ),
        {"mos": mo_ids},
    )
    conn.execute(sa.text("DELETE FROM work_orders WHERE mo_id = ANY(:mos)"), {"mos": mo_ids})
    conn.execute(sa.text("DELETE FROM mo_components WHERE mo_id = ANY(:mos)"), {"mos": mo_ids})
    conn.execute(sa.text("DELETE FROM manufacturing_orders WHERE id = ANY(:mos)"), {"mos": mo_ids})


def seed(tenant: str = "demo", clean_only: bool = False) -> dict:
    from maxxflow_data.engine import get_data_access

    data_access = get_data_access()
    with data_access.transaction(tenant=tenant) as conn:
        _clean(conn, tenant)
        if clean_only:
            return {"tenant": tenant, "removed": JOB_REFERENCES}

        md = _seed_master_data(conn, tenant)
        shared = _seed_shared(conn, tenant, md)
        operators = _seed_operator_history(conn, tenant, shared, md)
        for job in JOBS:
            _seed_job(conn, tenant, shared, md, operators, **job)

    return {"tenant": tenant, "jobs": JOB_REFERENCES, "operators": len(operators)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seed_m3_demo",
        description=(
            "Seed M3 production-delay demo jobs. `maxxflow seed --module m3_delay` does not "
            "write these raw tables — simulate_m3 emits features only."
        ),
    )
    parser.add_argument("--tenant", default="demo")
    parser.add_argument(
        "--clean", action="store_true", help="remove the seeded rows and exit"
    )
    args = parser.parse_args(argv)

    result = seed(args.tenant, clean_only=args.clean)
    if args.clean:
        print(f"removed M3 demo jobs from tenant={args.tenant}")
    else:
        print(f"seeded {len(result['jobs'])} M3 demo jobs into tenant={args.tenant}:")
        for reference in result["jobs"]:
            print(f"  {reference}")
    # CATEGORIES is imported for its side-effect-free vocabulary check: a
    # missing category here means the platform's master data moved and this
    # script would seed codes the rollup cannot resolve.
    assert "WORK_ORDER_STATUS" in CATEGORIES
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
