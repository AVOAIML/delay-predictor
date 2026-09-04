"""M2 feature SQL (plan §1, §1a). Reads via the DAL (search_path isolation);
NEVER selects items.rop_status (stale — derived live) or tenant_id."""

from __future__ import annotations

import pandas as pd

from maxxflow_data.engine import get_data_access
from maxxflow_data.masterdata_map import load_md_map

_ITEMS = """
SELECT id, item_name, part_number, item_type, unit_cost, unit_of_measurement, rop,
       default_warehouse_id, available_quantity, forecasted_quantity, status, custom_elements
FROM items WHERE deleted_at IS NULL
"""
_WAREHOUSES = """
SELECT id, warehouse_name, warehouse_type, status
FROM warehouses WHERE deleted_at IS NULL
"""
_MANUFACTURING_ORDERS = """
SELECT id, status_id, warehouse_id, scheduled_date, confirmed_at, completed_at,
       cancelled_at, created_at
FROM manufacturing_orders WHERE deleted_at IS NULL
"""
_MO_COMPONENTS = """
SELECT id, mo_id, item_id, product_id, component_type, required_qty, reserved_qty,
       consumed_qty, availability_id FROM mo_components
"""
_BOMS = "SELECT id, product_id FROM boms WHERE deleted_at IS NULL"
_BOM_COMPONENTS = """
SELECT id, bom_id, item_id, product_id, component_type, quantity
FROM bom_components
"""
_ITEM_VENDORS = "SELECT id, item_id, vendor_id, vendor_name, lead_time_days, unit_price FROM item_vendors"
# bill_created / bill_created_at are on purchase_orders (not the GRN) in schema.prisma;
# the GRN status-transition timestamp is updated_at.
_GRNS = """
SELECT id, reference_no, purchase_order_id, vendor_id, warehouse_id, scheduled_delivery_date,
       status, is_partially_closed, created_at, updated_at
FROM goods_received_notes
"""
_POS = """
SELECT id, reference_no, vendor_id, organisation_id, warehouse_id, scheduled_delivery_date, status
FROM purchase_orders
"""
_PO_LINES = """
SELECT id, purchase_order_id, item_id, item_name, ordered_quantity, received_quantity, unit_price
FROM purchase_order_lines
"""


def read_inventory_tables(tenant: str = "demo"):
    da = get_data_access()
    md = load_md_map(da, tenant)
    tables = {
        "items": da.query(_ITEMS, tenant=tenant),
        "warehouses": da.query(_WAREHOUSES, tenant=tenant),
        "manufacturing_orders": da.query(_MANUFACTURING_ORDERS, tenant=tenant),
        "mo_components": da.query(_MO_COMPONENTS, tenant=tenant),
        "boms": da.query(_BOMS, tenant=tenant),
        "bom_components": da.query(_BOM_COMPONENTS, tenant=tenant),
        "item_vendors": da.query(_ITEM_VENDORS, tenant=tenant),
        "goods_received_notes": da.query(_GRNS, tenant=tenant),
        "purchase_orders": da.query(_POS, tenant=tenant),
        "purchase_order_lines": da.query(_PO_LINES, tenant=tenant),
        "_meta": {"consumption_period_days": 7.0},
    }
    return tables, md
