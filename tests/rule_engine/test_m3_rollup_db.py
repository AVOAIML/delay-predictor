"""Integration test: build_job_rollups() against a REAL Postgres connection.

Everything under tests/unit is normally offline (see tests/conftest.py), but
this one is explicitly needs_db - it seeds one real job's raw rows straight
into the tenant_demo schema (the same tables m3_delay.dal.read_delay_tables
reads), runs the actual rollup builder against them, and asserts on the
result. It cleans up its own rows by id in a finally block, so it never
depends on - or clobbers - whatever else lives in tenant_demo.

Requires: `make bootstrap TENANT=demo` (or `docker compose -f compose.local.yml
up -d postgres` + `maxxflow db-provision --tenant demo`) beforehand. Without a
reachable DATA_DB_URL, conftest.py skips this automatically.
"""
from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.needs_db

TENANT = "demo"
JOB_REFERENCE = "TEST/M3-ROLLUP/00001"  # distinct from any real job reference


def _u() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def seeded_job():
    """Inserts one job (1 MO, 1 operation, 1 work order, 1 time log, 1
    component, its vendor, 1 PO, 1 GRN) with a job reference that cannot
    collide with real data, and deletes exactly those rows afterward."""
    from maxxflow_data.engine import get_data_access

    da = get_data_access()
    ids = {
        "mo": _u(), "op": _u(), "wc": _u(), "wo": _u(), "time_log": _u(),
        "operator": _u(), "item": _u(), "item_vendor": _u(), "vendor": _u(),
        "po": _u(), "po_line": _u(), "grn": _u(), "component": _u(),
        "cat_wos": _u(), "cat_optype": _u(), "cat_mostatus": _u(),
        "status_inprogress": _u(), "optype_independent": _u(), "mo_status": _u(),
        "avail_partial": _u(), "bom": _u(), "product": _u(), "warehouse_po": _u(),
        "warehouse_grn": _u(), "working_hours": _u(),
    }

    with da.transaction(tenant=TENANT) as conn:
        conn.execute(sa.text(f"""
            INSERT INTO master_data_category (id, name, code, status) VALUES
              ('{ids["cat_wos"]}', 'WORK_ORDER_STATUS_T', 'WORK_ORDER_STATUS_T', 'active'),
              ('{ids["cat_optype"]}', 'OPERATION_TYPE_T', 'OPERATION_TYPE_T', 'active'),
              ('{ids["cat_mostatus"]}', 'MO_STATUS_T', 'MO_STATUS_T', 'active')
        """))
        conn.execute(sa.text(f"""
            INSERT INTO master_data (id, category_id, name, code, status) VALUES
              ('{ids["status_inprogress"]}', '{ids["cat_wos"]}', 'In Progress', 'IN_PROGRESS', 'active'),
              ('{ids["optype_independent"]}', '{ids["cat_optype"]}', 'Independent', 'INDEPENDENT', 'active'),
              ('{ids["mo_status"]}', '{ids["cat_mostatus"]}', 'In Progress', 'IN_PROGRESS', 'active')
        """))
        conn.execute(sa.text(f"""
            INSERT INTO work_centers (id, name, code, working_hours_id)
            VALUES ('{ids["wc"]}', 'Test Cutting Bay', 'WC-TEST-{ids["wc"][:8]}', '{ids["working_hours"]}')
        """))
        conn.execute(sa.text(f"""
            INSERT INTO operations (id, bom_id, "operationName", "workCenterId", "estimatedDuration", operation_type_id, allowed_employees)
            VALUES ('{ids["op"]}', '{ids["bom"]}', 'Test Cutting', '{ids["wc"]}', 480, '{ids["optype_independent"]}', ARRAY['{ids["operator"]}']::uuid[])
        """))
        conn.execute(sa.text(f"""
            INSERT INTO manufacturing_orders (id, reference, product_id, quantity, status_id, component_status_id)
            VALUES ('{ids["mo"]}', '{JOB_REFERENCE}', '{ids["product"]}', 100, '{ids["mo_status"]}', '{ids["avail_partial"]}')
        """))
        conn.execute(sa.text(f"""
            INSERT INTO work_orders (id, mo_id, operation_id, work_center_id, quantity, units_done, expected_duration, real_duration, actual_start, assigned_operators, status_id)
            VALUES ('{ids["wo"]}', '{ids["mo"]}', '{ids["op"]}', '{ids["wc"]}', 100, 60, 480, 350, now() - interval '6 hours', ARRAY['{ids["operator"]}']::text[], '{ids["status_inprogress"]}')
        """))
        conn.execute(sa.text(f"""
            INSERT INTO work_order_time_logs (id, work_order_id, operator_id, started_at, ended_at, duration_minutes)
            VALUES ('{ids["time_log"]}', '{ids["wo"]}', '{ids["operator"]}', now() - interval '6 hours', now() - interval '10 minutes', 350)
        """))
        conn.execute(sa.text(f"""
            INSERT INTO items (id, item_name, part_number, available_quantity)
            VALUES ('{ids["item"]}', 'Test Steel Plate', 'TEST-STL', 40)
        """))
        conn.execute(sa.text(f"""
            INSERT INTO item_vendors (id, item_id, vendor_id, vendor_name, lead_time_days, unit_price)
            VALUES ('{ids["item_vendor"]}', '{ids["item"]}', '{ids["vendor"]}', 'Test Lanka Steel', 14, 12.50)
        """))
        conn.execute(sa.text(f"""
            INSERT INTO mo_components (id, mo_id, item_id, component_type, required_qty, reserved_qty, availability_id, purchase_order_id)
            VALUES ('{ids["component"]}', '{ids["mo"]}', '{ids["item"]}', 'item', 100, 40, '{ids["avail_partial"]}', '{ids["po"]}')
        """))
        conn.execute(sa.text(f"""
            INSERT INTO purchase_orders (id, reference_no, vendor_id, warehouse_id, scheduled_delivery_date, status, sent_at, created_at)
            VALUES ('{ids["po"]}', 'TEST-PO-5501', '{ids["vendor"]}', '{ids["warehouse_po"]}', '2026-08-15', 'Partially Received', '2026-08-01', '2026-07-30')
        """))
        conn.execute(sa.text(f"""
            INSERT INTO purchase_order_lines (id, purchase_order_id, item_id, item_name, ordered_quantity, received_quantity, unit_price)
            VALUES ('{ids["po_line"]}', '{ids["po"]}', '{ids["item"]}', 'Test Steel Plate', 100, 60, 12.50)
        """))
        conn.execute(sa.text(f"""
            INSERT INTO goods_received_notes (id, reference_no, purchase_order_id, vendor_id, warehouse_id, scheduled_delivery_date, status, is_partially_closed, created_at, updated_at)
            VALUES ('{ids["grn"]}', 'TEST-GRN-9001', '{ids["po"]}', '{ids["vendor"]}', '{ids["warehouse_grn"]}', '2026-08-15', 'Partially Received', true, '2026-08-16', '2026-08-18')
        """))

    try:
        yield ids
    finally:
        with da.transaction(tenant=TENANT) as conn:
            conn.execute(sa.text(f"DELETE FROM goods_received_notes WHERE id = '{ids['grn']}'"))
            conn.execute(sa.text(f"DELETE FROM purchase_order_lines WHERE id = '{ids['po_line']}'"))
            conn.execute(sa.text(f"DELETE FROM purchase_orders WHERE id = '{ids['po']}'"))
            conn.execute(sa.text(f"DELETE FROM mo_components WHERE id = '{ids['component']}'"))
            conn.execute(sa.text(f"DELETE FROM item_vendors WHERE id = '{ids['item_vendor']}'"))
            conn.execute(sa.text(f"DELETE FROM items WHERE id = '{ids['item']}'"))
            conn.execute(sa.text(f"DELETE FROM work_order_time_logs WHERE id = '{ids['time_log']}'"))
            conn.execute(sa.text(f"DELETE FROM work_orders WHERE id = '{ids['wo']}'"))
            conn.execute(sa.text(f"DELETE FROM manufacturing_orders WHERE id = '{ids['mo']}'"))
            conn.execute(sa.text(f"DELETE FROM operations WHERE id = '{ids['op']}'"))
            conn.execute(sa.text(f"DELETE FROM work_centers WHERE id = '{ids['wc']}'"))
            conn.execute(sa.text(
                f"DELETE FROM master_data WHERE id IN "
                f"('{ids['status_inprogress']}', '{ids['optype_independent']}', '{ids['mo_status']}')"
            ))
            conn.execute(sa.text(
                f"DELETE FROM master_data_category WHERE id IN "
                f"('{ids['cat_wos']}', '{ids['cat_optype']}', '{ids['cat_mostatus']}')"
            ))


def test_build_job_rollups_against_real_db(seeded_job):
    from m3_production_delay.rule_engine.dal import read_delay_tables
    from m3_production_delay.rule_engine.rollup import build_job_rollups

    tables, md = read_delay_tables(TENANT)
    rollups = build_job_rollups(tables, md, job_references=[JOB_REFERENCE])

    assert len(rollups) == 1
    job = rollups[0]
    assert job["job_id"] == JOB_REFERENCE
    assert len(job["operations"]) == 1

    op = job["operations"][0]
    assert op["operation_id"] == seeded_job["wo"]
    assert op["operation_type"] == "INDEPENDENT"
    assert op["status"] == "IN_PROGRESS"
    assert op["expected_duration_minutes"] == 480
    assert op["actual_duration_minutes"] == 350
    assert op["job_quantity"] == 100.0
    assert op["current_done_quantity"] == 60.0
    assert op["operator_count"] == 1

    operator = op["operators"][0]
    # PII discipline: the operator id in the result must be the HMAC token,
    # never the raw operator uuid that was inserted.
    assert operator["operator_id"] != seeded_job["operator"]
    assert operator["name"] is None
    assert len(operator["last_10_work_orders"]) == 1
    history = operator["last_10_work_orders"][0]
    assert history["work_order_id"] == seeded_job["wo"]
    assert history["elapsed_time_minutes"] == 350
    assert history["scheduled_time_minutes"] == 480
    assert history["operation_type"] == "INDEPENDENT"

    assert len(op["components"]) == 1
    component = op["components"][0]
    assert component["component_id"] == seeded_job["item"]
    assert component["name"] == "Test Steel Plate"
    assert component["required_quantity"] == 100.0
    assert component["available_quantity"] == 40.0

    vendor = component["vendor"]
    assert vendor is not None
    assert vendor["vendor_id"] == seeded_job["vendor"]
    assert vendor["name"] == "Test Lanka Steel"
    assert len(vendor["last_10_purchase_orders"]) == 1
    po = vendor["last_10_purchase_orders"][0]
    assert po["po_number"] == "TEST-PO-5501"
    # po_order_date is sent_at ?? created_at - sent_at was set, so it must win.
    assert str(po["po_order_date"]).startswith("2026-08-01")
    assert str(po["po_order_deadline"]).startswith("2026-08-15")
    assert str(po["grn_received_date"]).startswith("2026-08-18")


def test_build_job_rollups_returns_empty_for_unknown_job(seeded_job):
    """Sanity check that scoping by job_references actually filters -
    a reference that doesn't exist must yield no rollups, not everyone's."""
    from m3_production_delay.rule_engine.dal import read_delay_tables
    from m3_production_delay.rule_engine.rollup import build_job_rollups

    tables, md = read_delay_tables(TENANT)
    rollups = build_job_rollups(tables, md, job_references=["NO/SUCH/JOB"])
    assert rollups == []
