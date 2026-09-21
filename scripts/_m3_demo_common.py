"""Shared insert logic for the local-only M3 demo tooling
(`m3_demo_seed.py`, `m3_demo_api.py`). Not imported by anything under
`modules/` or `services/` — this exists purely to seed the real MRP tables
M3's rule engine reads (`modules/m3_production_delay/rule_engine/dal.py`),
since M3 has no synthetic generator of its own.

Every value is passed as a bound parameter, never interpolated into SQL —
unlike the fixed-scenario version this was split out of, `insert_job()` is
reachable from `m3_demo_api.py`'s POST endpoint with caller-supplied text
(product/operation/component names), which makes that the one place in this
demo tooling that sees untrusted input.

Master-data category/code rows use FIXED ids (not a fresh uuid per call) so
`ON CONFLICT (id) DO NOTHING` actually dedupes across repeated calls instead
of silently accumulating a new row every time — the bug the original
single-scenario seed script had (regenerating those ids on every run).
"""

from __future__ import annotations

import uuid

import sqlalchemy as sa

# Fixed, shared across every job this tooling ever creates.
CAT_WORK_ORDER_STATUS = "00000000-0000-4000-8000-000000000001"
CAT_OPERATION_TYPE = "00000000-0000-4000-8000-000000000002"
CAT_MO_STATUS = "00000000-0000-4000-8000-000000000003"
STATUS_IN_PROGRESS = "00000000-0000-4000-8000-000000000011"
OPTYPE_INDEPENDENT = "00000000-0000-4000-8000-000000000012"
MO_STATUS_IN_PROGRESS = "00000000-0000-4000-8000-000000000013"
AVAILABILITY_PARTIAL = "00000000-0000-4000-8000-000000000014"


def _u() -> str:
    return str(uuid.uuid4())


def _insert_operator_history(
    conn: sa.Connection,
    *,
    job_reference: str,
    operation_id: str,
    work_center_id: str,
    operator_id: str,
    product_id: str,
) -> None:
    """Seed three completed jobs for the assigned demo operator.

    Their elapsed/scheduled ratios are 1.3, 1.4 and 1.5, so the rule
    engine's operator pace input is deterministically 1.4.
    """
    for index, elapsed_minutes in enumerate((130, 140, 150), start=1):
        history_mo_id = _u()
        history_wo_id = _u()
        conn.execute(
            sa.text(
                """
                INSERT INTO manufacturing_orders
                    (id, reference, product_id, quantity, status_id, component_status_id,
                     confirmed_at, completed_at)
                VALUES
                    (:id, :reference, :product, 1, :status, :avail,
                     now() - make_interval(days => :days_old),
                     now() - make_interval(days => :days_old) + make_interval(mins => :elapsed))
                """
            ),
            {
                "id": history_mo_id,
                "reference": f"{job_reference}-H{index}",
                "product": product_id,
                "status": MO_STATUS_IN_PROGRESS,
                "avail": AVAILABILITY_PARTIAL,
                "days_old": index + 3,
                "elapsed": elapsed_minutes,
            },
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO work_orders
                    (id, mo_id, operation_id, work_center_id, quantity, units_done,
                     expected_duration, real_duration, scheduled_start, scheduled_end,
                     actual_start, actual_end, assigned_operators, status_id)
                VALUES
                    (:id, :mo, :op, :wc, 1, 1, 100, :elapsed,
                     now() - make_interval(days => :days_old),
                     now() - make_interval(days => :days_old) + interval '100 minutes',
                     now() - make_interval(days => :days_old),
                     now() - make_interval(days => :days_old) + make_interval(mins => :elapsed),
                     ARRAY[:operator]::text[], :status)
                """
            ),
            {
                "id": history_wo_id,
                "mo": history_mo_id,
                "op": operation_id,
                "wc": work_center_id,
                "elapsed": elapsed_minutes,
                "days_old": index + 3,
                "operator": operator_id,
                "status": STATUS_IN_PROGRESS,
            },
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO work_order_time_logs
                    (id, work_order_id, operator_id, started_at, ended_at, duration_minutes)
                VALUES
                    (:id, :wo, :operator,
                     now() - make_interval(days => :days_old),
                     now() - make_interval(days => :days_old) + make_interval(mins => :elapsed),
                     :elapsed)
                """
            ),
            {
                "id": _u(),
                "wo": history_wo_id,
                "operator": operator_id,
                "days_old": index + 3,
                "elapsed": elapsed_minutes,
            },
        )


def _insert_supplier_history(
    conn: sa.Connection,
    *,
    job_reference: str,
    item_id: str,
    item_name: str,
    component_index: int,
) -> None:
    """Seed a vendor and three received POs for one component.

    Each PO promises a ten-day window and arrives after 13, 15 or 17 days,
    producing lead-time ratios 1.3, 1.5 and 1.7 (mean 1.5).
    """
    vendor_id = _u()
    conn.execute(
        sa.text(
            """
            INSERT INTO item_vendors
                (id, item_id, vendor_id, vendor_name, lead_time_days, unit_price)
            VALUES (:id, :item, :vendor, :name, 10, 1.00)
            """
        ),
        {
            "id": _u(),
            "item": item_id,
            "vendor": vendor_id,
            "name": f"Demo Supplier {component_index}",
        },
    )

    for po_index, received_days in enumerate((13, 15, 17), start=1):
        po_id = _u()
        warehouse_id = _u()
        sent_days_ago = 40 + po_index
        conn.execute(
            sa.text(
                """
                INSERT INTO purchase_orders
                    (id, reference_no, vendor_id, warehouse_id,
                     scheduled_delivery_date, status, sent_at, created_at)
                VALUES
                    (:id, :reference, :vendor, :warehouse,
                     now() - make_interval(days => :sent_days_ago) + interval '10 days',
                     'Goods Received',
                     now() - make_interval(days => :sent_days_ago),
                     now() - make_interval(days => :sent_days_ago) - interval '1 day')
                """
            ),
            {
                "id": po_id,
                "reference": f"DEMO-PO-{job_reference[-5:]}-{component_index}-{po_index}",
                "vendor": vendor_id,
                "warehouse": warehouse_id,
                "sent_days_ago": sent_days_ago,
            },
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO purchase_order_lines
                    (id, purchase_order_id, item_id, item_name,
                     ordered_quantity, received_quantity, unit_price)
                VALUES (:id, :po, :item, :name, 100, 100, 1.00)
                """
            ),
            {"id": _u(), "po": po_id, "item": item_id, "name": item_name},
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO goods_received_notes
                    (id, reference_no, purchase_order_id, vendor_id, warehouse_id,
                     scheduled_delivery_date, status, is_partially_closed,
                     created_at, updated_at)
                VALUES
                    (:id, :reference, :po, :vendor, :warehouse,
                     now() - make_interval(days => :sent_days_ago) + interval '10 days',
                     'Goods Received', false,
                     now() - make_interval(days => :sent_days_ago) + make_interval(days => :received_days),
                     now() - make_interval(days => :sent_days_ago) + make_interval(days => :received_days))
                """
            ),
            {
                "id": _u(),
                "reference": f"DEMO-GRN-{job_reference[-5:]}-{component_index}-{po_index}",
                "po": po_id,
                "vendor": vendor_id,
                "warehouse": warehouse_id,
                "sent_days_ago": sent_days_ago,
                "received_days": received_days,
            },
        )


def ensure_reference_data(conn: sa.Connection) -> None:
    """Idempotent: the fixed master-data rows every seeded job points at."""
    conn.execute(
        sa.text(
            """
            INSERT INTO master_data_category (id, name, code, status) VALUES
              (:cat_wos, 'WORK_ORDER_STATUS_T', 'WORK_ORDER_STATUS_T', 'active'),
              (:cat_optype, 'OPERATION_TYPE_T', 'OPERATION_TYPE_T', 'active'),
              (:cat_mostatus, 'MO_STATUS_T', 'MO_STATUS_T', 'active')
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {"cat_wos": CAT_WORK_ORDER_STATUS, "cat_optype": CAT_OPERATION_TYPE, "cat_mostatus": CAT_MO_STATUS},
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO master_data (id, category_id, name, code, status) VALUES
              (:status_inprogress, :cat_wos, 'In Progress', 'IN_PROGRESS', 'active'),
              (:optype_independent, :cat_optype, 'Independent', 'INDEPENDENT', 'active'),
              (:mo_status, :cat_mostatus, 'In Progress', 'IN_PROGRESS', 'active'),
              (:avail_partial, :cat_mostatus, 'Partially Available', 'PARTIAL', 'active')
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {
            "status_inprogress": STATUS_IN_PROGRESS,
            "optype_independent": OPTYPE_INDEPENDENT,
            "mo_status": MO_STATUS_IN_PROGRESS,
            "avail_partial": AVAILABILITY_PARTIAL,
            "cat_wos": CAT_WORK_ORDER_STATUS,
            "cat_optype": CAT_OPERATION_TYPE,
            "cat_mostatus": CAT_MO_STATUS,
        },
    )


def insert_job(
    conn: sa.Connection,
    *,
    job_reference: str,
    quantity: float,
    operation_name: str,
    expected_duration_minutes: float,
    actual_duration_minutes: float,
    components: list[dict],  # [{"name": str, "required_quantity": float, "available_quantity": float}]
) -> None:
    """Inserts one job — one MO, one operation, one work order actually
    logging `actual_duration_minutes` against `expected_duration_minutes`,
    and one mo_components row per entry in `components` — using exactly the
    columns modules/m3_production_delay/rule_engine/dal.py reads. Call
    `ensure_reference_data(conn)` on the same connection first.

    Runs inside the caller's transaction; nothing here commits on its own.
    """
    ids = {
        name: _u()
        for name in ("mo", "op", "wc", "wo", "time_log", "operator", "product", "bom", "working_hours")
    }

    conn.execute(
        sa.text(
            """
            INSERT INTO work_centers (id, name, code, working_hours_id)
            VALUES (:id, 'Demo Work Center', :code, :working_hours)
            """
        ),
        {"id": ids["wc"], "code": f"WC-DEMO-{ids['wc'][:8]}", "working_hours": ids["working_hours"]},
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO operations (id, bom_id, "operationName", "workCenterId", "estimatedDuration", operation_type_id, allowed_employees)
            VALUES (:id, :bom, :name, :wc, :duration, :optype, ARRAY[CAST(:operator AS uuid)])
            """
        ),
        {
            "id": ids["op"], "bom": ids["bom"], "name": operation_name, "wc": ids["wc"],
            "duration": expected_duration_minutes, "optype": OPTYPE_INDEPENDENT, "operator": ids["operator"],
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO manufacturing_orders (id, reference, product_id, quantity, status_id, component_status_id)
            VALUES (:id, :reference, :product, :quantity, :status, :avail)
            """
        ),
        {
            "id": ids["mo"], "reference": job_reference, "product": ids["product"],
            "quantity": quantity, "status": MO_STATUS_IN_PROGRESS, "avail": AVAILABILITY_PARTIAL,
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO work_orders (id, mo_id, operation_id, work_center_id, quantity, units_done, expected_duration, real_duration, actual_start, assigned_operators, status_id)
            VALUES (:id, :mo, :op, :wc, :quantity, :quantity, :expected, :actual, now() - make_interval(mins => CAST(:actual AS integer)), ARRAY[:operator]::text[], :status)
            """
        ),
        {
            "id": ids["wo"], "mo": ids["mo"], "op": ids["op"], "wc": ids["wc"], "quantity": quantity,
            "expected": expected_duration_minutes, "actual": actual_duration_minutes,
            "operator": ids["operator"], "status": STATUS_IN_PROGRESS,
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO work_order_time_logs (id, work_order_id, operator_id, started_at, ended_at, duration_minutes)
            VALUES (:id, :wo, :operator, now() - make_interval(mins => CAST(:actual AS integer)), now(), :actual)
            """
        ),
        {"id": _u(), "wo": ids["wo"], "operator": ids["operator"], "actual": actual_duration_minutes},
    )

    _insert_operator_history(
        conn,
        job_reference=job_reference,
        operation_id=ids["op"],
        work_center_id=ids["wc"],
        operator_id=ids["operator"],
        product_id=ids["product"],
    )

    for component_index, component in enumerate(components, start=1):
        item_id = _u()
        conn.execute(
            sa.text(
                """
                INSERT INTO items (id, item_name, part_number, available_quantity)
                VALUES (:id, :name, :part_number, :available)
                """
            ),
            {
                "id": item_id, "name": component["name"], "part_number": f"DEMO-{item_id[:8].upper()}",
                "available": component["available_quantity"],
            },
        )
        conn.execute(
            sa.text(
                """
                INSERT INTO mo_components (id, mo_id, item_id, component_type, required_qty, reserved_qty, availability_id)
                VALUES (:id, :mo, :item, 'item', :required, :available, :avail)
                """
            ),
            {
                "id": _u(), "mo": ids["mo"], "item": item_id, "required": component["required_quantity"],
                "available": component["available_quantity"], "avail": AVAILABILITY_PARTIAL,
            },
        )
        _insert_supplier_history(
            conn,
            job_reference=job_reference,
            item_id=item_id,
            item_name=component["name"],
            component_index=component_index,
        )
