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
STATUS_NOT_STARTED = "00000000-0000-4000-8000-000000000015"
OPTYPE_DEPENDENT = "00000000-0000-4000-8000-000000000016"
CAT_PRODUCT_TYPE = "00000000-0000-4000-8000-000000000004"
PRODUCT_TYPE_MANUFACTURED = "00000000-0000-4000-8000-000000000017"
DEMO_WAREHOUSE = "00000000-0000-4000-8000-000000000021"
DEMO_CATALOG_PRODUCT = "00000000-0000-4000-8000-000000000022"
DEMO_CATALOG_BOM = "00000000-0000-4000-8000-000000000023"
DEMO_CATALOG_PANEL = "00000000-0000-4000-8000-000000000024"
DEMO_CATALOG_FASTENER = "00000000-0000-4000-8000-000000000025"
DEMO_CATALOG_WORK_CENTER = "00000000-0000-4000-8000-000000000026"
DEMO_CATALOG_CUT_OP = "00000000-0000-4000-8000-000000000027"
DEMO_CATALOG_ASSEMBLY_OP = "00000000-0000-4000-8000-000000000028"


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
              (:cat_mostatus, 'MO_STATUS_T', 'MO_STATUS_T', 'active'),
              (:cat_product, 'PRODUCT_TYPE_T', 'PRODUCT_TYPE_T', 'active')
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {
            "cat_wos": CAT_WORK_ORDER_STATUS,
            "cat_optype": CAT_OPERATION_TYPE,
            "cat_mostatus": CAT_MO_STATUS,
            "cat_product": CAT_PRODUCT_TYPE,
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO master_data (id, category_id, name, code, status) VALUES
              (:status_inprogress, :cat_wos, 'In Progress', 'IN_PROGRESS', 'active'),
              (:status_not_started, :cat_wos, 'Not Started', 'NOT_STARTED', 'active'),
              (:optype_independent, :cat_optype, 'Independent', 'INDEPENDENT', 'active'),
              (:optype_dependent, :cat_optype, 'Dependent', 'DEPENDENT', 'active'),
              (:mo_status, :cat_mostatus, 'In Progress', 'IN_PROGRESS', 'active'),
              (:avail_partial, :cat_mostatus, 'Partially Available', 'PARTIAL', 'active'),
              (:product_type, :cat_product, 'Manufactured', 'MANUFACTURED', 'active')
            ON CONFLICT (id) DO NOTHING
            """
        ),
        {
            "status_inprogress": STATUS_IN_PROGRESS,
            "status_not_started": STATUS_NOT_STARTED,
            "optype_independent": OPTYPE_INDEPENDENT,
            "optype_dependent": OPTYPE_DEPENDENT,
            "mo_status": MO_STATUS_IN_PROGRESS,
            "avail_partial": AVAILABILITY_PARTIAL,
            "cat_wos": CAT_WORK_ORDER_STATUS,
            "cat_optype": CAT_OPERATION_TYPE,
            "cat_mostatus": CAT_MO_STATUS,
            "cat_product": CAT_PRODUCT_TYPE,
            "product_type": PRODUCT_TYPE_MANUFACTURED,
        },
    )
    conn.execute(sa.text("""
        INSERT INTO warehouses (id, warehouse_name, short_name)
        VALUES (:id, 'Demo Main Warehouse', 'DEMO') ON CONFLICT (id) DO NOTHING
    """), {"id": DEMO_WAREHOUSE})
    conn.execute(sa.text("""
        INSERT INTO items (id, item_name, part_number, available_quantity) VALUES
          (:panel, 'Desk Panel', 'DEMO-DESK-PANEL', 140),
          (:fastener, 'Desk Fastener Set', 'DEMO-DESK-FASTENER', 600)
        ON CONFLICT (id) DO NOTHING
    """), {"panel": DEMO_CATALOG_PANEL, "fastener": DEMO_CATALOG_FASTENER})
    conn.execute(sa.text("""
        INSERT INTO products
          (id, sku, name, product_type_id, sales_price, unit_cost,
           default_warehouse_id, project_type, route)
        VALUES
          (:id, 'DEMO-OFFICE-DESK', 'Office Desk', :type, 500, 250,
           :warehouse, 'catalog_item', 'manufacture')
        ON CONFLICT (id) DO NOTHING
    """), {
        "id": DEMO_CATALOG_PRODUCT, "type": PRODUCT_TYPE_MANUFACTURED,
        "warehouse": DEMO_WAREHOUSE,
    })
    conn.execute(sa.text("""
        INSERT INTO boms (id, code, name, product_id, version)
        VALUES (:id, 'BOM-DEMO-DESK', 'Office Desk BoM', :product, '1.0')
        ON CONFLICT (id) DO NOTHING
    """), {"id": DEMO_CATALOG_BOM, "product": DEMO_CATALOG_PRODUCT})
    conn.execute(sa.text("""
        INSERT INTO bom_components
          (id, bom_id, item_id, component_type, quantity) VALUES
          ('00000000-0000-4000-8000-000000000031', :bom, :panel, 'item', 4),
          ('00000000-0000-4000-8000-000000000032', :bom, :fastener, 'item', 1)
        ON CONFLICT (id) DO NOTHING
    """), {
        "bom": DEMO_CATALOG_BOM, "panel": DEMO_CATALOG_PANEL,
        "fastener": DEMO_CATALOG_FASTENER,
    })
    conn.execute(sa.text("""
        INSERT INTO work_centers (id, name, code, working_hours_id)
        VALUES (:id, 'Desk Production Cell', 'WC-DEMO-DESK', :hours)
        ON CONFLICT (id) DO NOTHING
    """), {"id": DEMO_CATALOG_WORK_CENTER, "hours": DEMO_CATALOG_WORK_CENTER})
    conn.execute(sa.text("""
        INSERT INTO operations
          (id, bom_id, "operationName", "workCenterId", "estimatedDuration",
           operation_type_id, allowed_employees) VALUES
          (:cut, :bom, 'Cut Desk Panels', :wc, 180, :independent, ARRAY[]::uuid[]),
          (:assembly, :bom, 'Assemble Desk', :wc, 240, :dependent, ARRAY[]::uuid[])
        ON CONFLICT (id) DO NOTHING
    """), {
        "cut": DEMO_CATALOG_CUT_OP, "assembly": DEMO_CATALOG_ASSEMBLY_OP,
        "bom": DEMO_CATALOG_BOM, "wc": DEMO_CATALOG_WORK_CENTER,
        "independent": OPTYPE_INDEPENDENT, "dependent": OPTYPE_DEPENDENT,
    })
    conn.execute(sa.text("""
        INSERT INTO operation_dependencies (id, operation_id, depends_on_id)
        VALUES ('00000000-0000-4000-8000-000000000033', :assembly, :cut)
        ON CONFLICT (id) DO NOTHING
    """), {"assembly": DEMO_CATALOG_ASSEMBLY_OP, "cut": DEMO_CATALOG_CUT_OP})


def insert_job(
    conn: sa.Connection,
    *,
    job_reference: str,
    product_name: str,
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
            INSERT INTO manufacturing_orders
                (id, reference, product_id, quantity, scheduled_date, status_id,
                 component_status_id, custom_elements)
            VALUES
                (:id, :reference, :product, :quantity, CURRENT_DATE, :status, :avail,
                 jsonb_build_object('demo_product', CAST(:product_name AS text),
                                    'demo_bom', CAST(:product_name AS text) || ' BoM'))
            """
        ),
        {
            "id": ids["mo"], "reference": job_reference, "product": ids["product"],
            "product_name": product_name, "quantity": quantity,
            "status": MO_STATUS_IN_PROGRESS, "avail": AVAILABILITY_PARTIAL,
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


def insert_configured_job(
    conn: sa.Connection,
    *,
    job_reference: str,
    product_id: str,
    product_name: str,
    bom_id: str,
    bom_name: str,
    quantity: float,
    components: list[dict],
    work_orders: list[dict],
) -> None:
    """Insert the multi-operation MO assembled by the local frontend form."""
    mo_id = _u()
    work_center_id = _u()
    operator_id = _u()
    conn.execute(sa.text("""
        INSERT INTO work_centers (id, name, code, working_hours_id)
        VALUES (:id, 'Frontend MO Work Center', :code, :hours)
    """), {
        "id": work_center_id, "code": f"WC-FORM-{work_center_id[:8]}",
        "hours": work_center_id,
    })
    conn.execute(sa.text("""
        INSERT INTO manufacturing_orders
          (id, reference, product_id, bom_id, quantity, scheduled_date, status_id,
           component_status_id, confirmed_at, custom_elements)
        VALUES
          (:id, :reference, :product, :bom, :quantity, CURRENT_DATE, :status,
           :availability, now(),
           jsonb_build_object('demo_product', CAST(:product_name AS text),
                              'demo_bom', CAST(:bom_name AS text)))
    """), {
        "id": mo_id, "reference": job_reference, "product": product_id,
        "bom": bom_id, "quantity": quantity, "status": MO_STATUS_IN_PROGRESS,
        "availability": AVAILABILITY_PARTIAL, "product_name": product_name,
        "bom_name": bom_name,
    })

    for component in components:
        item_id = _u()
        conn.execute(sa.text("""
            INSERT INTO items (id, item_name, part_number, available_quantity)
            VALUES (:id, :name, :part, :available)
        """), {
            "id": item_id, "name": component["name"],
            "part": f"FORM-{item_id[:8].upper()}",
            "available": component["available_quantity"],
        })
        conn.execute(sa.text("""
            INSERT INTO mo_components
              (id, mo_id, item_id, component_type, required_qty, reserved_qty,
               availability_id)
            VALUES (:id, :mo, :item, 'item', :required, 0, :availability)
        """), {
            "id": _u(), "mo": mo_id, "item": item_id,
            "required": component["required_quantity"],
            "availability": AVAILABILITY_PARTIAL,
        })

    operation_ids: list[str] = []
    for index, work_order in enumerate(work_orders):
        operation_id = work_order.get("operation_id") or _u()
        operation_work_center_id = work_order.get("work_center_id") or work_center_id
        operation_ids.append(operation_id)
        if not work_order.get("operation_id"):
            conn.execute(sa.text("""
                INSERT INTO operations
                  (id, bom_id, "operationName", "workCenterId", "estimatedDuration",
                   operation_type_id, allowed_employees)
                VALUES (:id, :bom, :name, :wc, :duration, :type, ARRAY[]::uuid[])
            """), {
                "id": operation_id, "bom": bom_id, "name": work_order["name"],
                "wc": operation_work_center_id,
                "duration": round(work_order["expected_duration_hours"] * 60),
                "type": (
                    OPTYPE_DEPENDENT
                    if work_order.get("depends_on_index") is not None
                    else OPTYPE_INDEPENDENT
                ),
            })

    for index, work_order in enumerate(work_orders):
        operation_work_center_id = work_order.get("work_center_id") or work_center_id
        predecessor_index = work_order.get("depends_on_index")
        if predecessor_index is not None and not work_order.get("operation_id"):
            conn.execute(sa.text("""
                INSERT INTO operation_dependencies (id, operation_id, depends_on_id)
                VALUES (:id, :operation, :predecessor)
                ON CONFLICT (operation_id, depends_on_id) DO NOTHING
            """), {
                "id": _u(), "operation": operation_ids[index],
                "predecessor": operation_ids[predecessor_index],
            })

        expected_minutes = round(work_order["expected_duration_hours"] * 60)
        actual_hours = work_order.get("actual_duration_hours")
        actual_minutes = None if actual_hours is None else round(actual_hours * 60)
        units_done = work_order.get("units_done", 0)
        started = actual_minutes is not None and actual_minutes > 0
        wo_id = _u()
        conn.execute(sa.text("""
            INSERT INTO work_orders
              (id, mo_id, operation_id, work_center_id, quantity, units_done,
               expected_duration, real_duration, scheduled_start, scheduled_end,
               actual_start, assigned_operators, status_id)
            VALUES
              (:id, :mo, :operation, :wc, :quantity, :done, :expected, :actual,
               now(), now() + make_interval(mins => :expected),
               CASE WHEN :started THEN now() - make_interval(mins => :actual) ELSE NULL END,
               CASE WHEN :started THEN ARRAY[:operator]::text[] ELSE ARRAY[]::text[] END,
               :status)
        """), {
            "id": wo_id, "mo": mo_id, "operation": operation_ids[index],
            "wc": operation_work_center_id, "quantity": quantity, "done": units_done,
            "expected": expected_minutes, "actual": actual_minutes,
            "started": started, "operator": operator_id,
            "status": STATUS_IN_PROGRESS if started else STATUS_NOT_STARTED,
        })
        if started:
            conn.execute(sa.text("""
                INSERT INTO work_order_time_logs
                  (id, work_order_id, operator_id, started_at, ended_at, duration_minutes)
                VALUES (:id, :wo, :operator,
                        now() - make_interval(mins => :actual), now(), :actual)
            """), {
                "id": _u(), "wo": wo_id, "operator": operator_id,
                "actual": actual_minutes,
            })


def insert_cascading_delay_job(conn: sa.Connection, *, job_reference: str) -> None:
    """Seed the user-story cascading-delay scenario into the real MRP tables.

    The independent predecessor is 25% complete by quantity (25 / 100), with
    78 minutes logged against a 240-minute full-operation plan. The rule
    engine therefore computes::

        work_done_percentage = 0.25
        time_overrun_ratio = 78 / (0.25 * 240) = 1.30

    The dependent operation has not started and is linked to the predecessor
    through ``operation_dependencies``. The parent MO's scheduled date is two
    days in the past so the seed also carries the story's scheduled-overrun
    condition, although the current rule engine does not score that field.

    Call ``ensure_reference_data(conn)`` on the same connection first.
    """
    ids = {
        name: _u()
        for name in (
            "mo", "predecessor_op", "dependent_op", "wc", "predecessor_wo",
            "dependent_wo", "operator", "product", "bom", "working_hours",
            "panel_item", "fastener_item",
        )
    }

    conn.execute(
        sa.text(
            """
            INSERT INTO work_centers (id, name, code, working_hours_id)
            VALUES (:id, 'Cascade Demo Work Center', :code, :working_hours)
            """
        ),
        {
            "id": ids["wc"],
            "code": f"WC-CASCADE-{ids['wc'][:8]}",
            "working_hours": ids["working_hours"],
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO items (id, item_name, part_number, available_quantity)
            VALUES
                (:panel_item, 'Cut Panel Blanks', :panel_part, 120),
                (:fastener_item, 'Assembly Fasteners', :fastener_part, 500)
            """
        ),
        {
            "panel_item": ids["panel_item"],
            "panel_part": f"CASCADE-PANEL-{ids['panel_item'][:8].upper()}",
            "fastener_item": ids["fastener_item"],
            "fastener_part": f"CASCADE-FAST-{ids['fastener_item'][:8].upper()}",
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO operations
                (id, bom_id, "operationName", "workCenterId", "estimatedDuration",
                 operation_type_id, allowed_employees)
            VALUES
                (:predecessor, :bom, 'Cut Components', :wc, 240,
                 :independent, ARRAY[CAST(:operator AS uuid)]),
                (:dependent, :bom, 'Final Assembly', :wc, 180,
                 :dependent_type, ARRAY[]::uuid[])
            """
        ),
        {
            "predecessor": ids["predecessor_op"],
            "dependent": ids["dependent_op"],
            "bom": ids["bom"],
            "wc": ids["wc"],
            "independent": OPTYPE_INDEPENDENT,
            "dependent_type": OPTYPE_DEPENDENT,
            "operator": ids["operator"],
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO operation_dependencies (id, operation_id, depends_on_id)
            VALUES (:id, :dependent, :predecessor)
            """
        ),
        {"id": _u(), "dependent": ids["dependent_op"], "predecessor": ids["predecessor_op"]},
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO manufacturing_orders
                (id, reference, product_id, bom_id, quantity, scheduled_date,
                 status_id, component_status_id, confirmed_at, custom_elements)
            VALUES
                (:id, :reference, :product, :bom, 100,
                 CURRENT_DATE - interval '2 days', :status, :avail,
                 now() - interval '3 days',
                 jsonb_build_object('demo_product', 'Critical-path Assembly',
                                    'demo_bom', 'Critical-path Assembly BoM'))
            """
        ),
        {
            "id": ids["mo"],
            "reference": job_reference,
            "product": ids["product"],
            "bom": ids["bom"],
            "status": MO_STATUS_IN_PROGRESS,
            "avail": AVAILABILITY_PARTIAL,
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO mo_components
                (id, mo_id, item_id, component_type, required_qty, reserved_qty,
                 consumed_qty, availability_id)
            VALUES
                (:panel_component, :mo, :panel_item, 'item', 100, 100, 25, :available),
                (:fastener_component, :mo, :fastener_item, 'item', 400, 400, 0, :available)
            """
        ),
        {
            "panel_component": _u(),
            "fastener_component": _u(),
            "mo": ids["mo"],
            "panel_item": ids["panel_item"],
            "fastener_item": ids["fastener_item"],
            "available": AVAILABILITY_PARTIAL,
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
                (:predecessor_wo, :mo, :predecessor_op, :wc, 100, 25,
                 240, 78, now() - interval '2 days',
                 now() - interval '2 days' + interval '240 minutes',
                 now() - interval '78 minutes', NULL,
                 ARRAY[:operator]::text[], :in_progress),
                (:dependent_wo, :mo, :dependent_op, :wc, 100, 0,
                 180, NULL, now() - interval '1 day',
                 now() - interval '1 day' + interval '180 minutes',
                 NULL, NULL, ARRAY[]::text[], :not_started)
            """
        ),
        {
            "predecessor_wo": ids["predecessor_wo"],
            "dependent_wo": ids["dependent_wo"],
            "mo": ids["mo"],
            "predecessor_op": ids["predecessor_op"],
            "dependent_op": ids["dependent_op"],
            "wc": ids["wc"],
            "operator": ids["operator"],
            "in_progress": STATUS_IN_PROGRESS,
            "not_started": STATUS_NOT_STARTED,
        },
    )
    conn.execute(
        sa.text(
            """
            INSERT INTO work_order_time_logs
                (id, work_order_id, operator_id, started_at, ended_at, duration_minutes)
            VALUES
                (:id, :wo, :operator, now() - interval '78 minutes', now(), 78)
            """
        ),
        {"id": _u(), "wo": ids["predecessor_wo"], "operator": ids["operator"]},
    )

    _insert_operator_history(
        conn,
        job_reference=job_reference,
        operation_id=ids["predecessor_op"],
        work_center_id=ids["wc"],
        operator_id=ids["operator"],
        product_id=ids["product"],
    )
