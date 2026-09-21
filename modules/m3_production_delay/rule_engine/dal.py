"""M3 feature SQL — job/operation/operator/component delay-risk reads.

Same shape as ``m1_quote.dal`` / ``m2_inventory.dal``: plain SELECTs run through
the DAL (search_path isolation, ``tenant_id``/``rop_status`` guard). Column sets
are restricted to what ``schema_def.TENANT_TABLES`` actually provisions locally
— two tables from the full ``schema.prisma`` (``bom_operation_orders``,
``organisation_item_mappings``) are NOT in that local subset, so this module
cannot read them; see the AVAILABLE / PARTIAL / NOT AVAILABLE notes below.

AVAILABLE (real columns, read as-is):
  - job_id                    -> manufacturing_orders.reference
  - job_quantity               -> work_orders.quantity (per-operation quantity)
  - operation_id                -> work_orders.id
  - operation_type              -> operations.operation_type_id -> master_data
  - status                      -> work_orders.status_id -> master_data
  - expected_duration_minutes  -> work_orders.expected_duration
  - actual_duration_minutes    -> work_orders.real_duration
  - current_done_quantity      -> work_orders.units_done
  - operator history rows      -> work_order_time_logs (started_at/ended_at/duration_minutes)
  - component required/available -> mo_components.required_qty, items.available_quantity
  - vendor for a component     -> item_vendors (vendor_id, vendor_name) — the ONLY
                                   vendor source in this module's table subset;
                                   ``organisation_item_mappings`` is not provisioned
                                   locally, so there is no "is_primary" signal here
  - po_number / po_order_deadline -> purchase_orders.reference_no / scheduled_delivery_date

PARTIAL (derivable but approximated — never treat as ground truth):
  - operator_count             -> len(assigned_operators); not a stored column
  - po_order_date               -> purchase_orders.sent_at, falling back to created_at
                                   (no explicit "order placed" column exists)
  - GRN "received" timestamp    -> goods_received_notes.updated_at, gated on
                                   status IN ('Goods Received','Partially Received')
                                   via transforms.grn_on_time — the established
                                   status-transition proxy used by M2 (see
                                   NOTES.md / plan §12a #4). NOT a true received-date
                                   column; do not expose it as one.

NOT AVAILABLE in this module (would need columns/tables outside the local
AI/ML schema subset — see schema_def.py's scoping note):
  - sequence_no                 -> would come from bom_operation_orders.position,
                                    which is not part of TENANT_TABLES here
  - vendor "is_primary" flag    -> would come from organisation_item_mappings,
                                    also not part of TENANT_TABLES here

Operator UUIDs are PII and must never leave this module raw: every
``operator_id`` / ``assigned_operators`` value is HMAC-pseudonymized (plan
§1a operator row, §12a #9) before the frame is returned, using
``maxxflow_data.transforms.pseudonymize_operators``. Downstream code must key
off the pseudonymous token, never the raw uuid.
"""

from __future__ import annotations

import pandas as pd

from maxxflow_core.settings import get_settings
from maxxflow_data.engine import get_data_access
from maxxflow_data.masterdata_map import load_md_map
from maxxflow_data.transforms import pseudonymize_operators

_MANUFACTURING_ORDERS = """
SELECT id, reference, product_id, bom_id, quantity, scheduled_date,
       status_id, component_status_id, confirmed_at, completed_at, cancelled_at
FROM manufacturing_orders WHERE deleted_at IS NULL
"""

_OPERATIONS = """
SELECT id, bom_id, "operationName" AS operation_name, "workCenterId" AS work_center_id,
       "estimatedDuration" AS estimated_duration, operation_type_id, allowed_employees
FROM operations WHERE deleted_at IS NULL
"""

_OPERATION_DEPENDENCIES = """
SELECT id, operation_id, depends_on_id FROM operation_dependencies
"""

_WORK_CENTERS = """
SELECT id, name, code, setup_time, cleanup_time, cost_per_hour,
       employee_cost_per_hour, allowed_employees
FROM work_centers WHERE deleted_at IS NULL
"""

# job_id / job_quantity / operation_id / status / expected_duration_minutes /
# actual_duration_minutes / current_done_quantity all come from this table.
_WORK_ORDERS = """
SELECT id, mo_id, operation_id, work_center_id, quantity, units_done,
       expected_duration, real_duration, scheduled_start, scheduled_end,
       actual_start, actual_end, assigned_operators, status_id, created_at
FROM work_orders
"""

# per-operator elapsed time — WorkOrder.real_duration is the combined total
# across every assigned operator, so an individual operator's own minutes
# on a WO must be read from here, not from work_orders.real_duration.
_WORK_ORDER_TIME_LOGS = """
SELECT id, work_order_id, operator_id, started_at, ended_at, duration_minutes
FROM work_order_time_logs
"""

_MO_COMPONENTS = """
SELECT id, mo_id, item_id, product_id, component_type, required_qty,
       reserved_qty, consumed_qty, availability_id, purchase_order_id
FROM mo_components
"""

_ITEMS = """
SELECT id, item_name, part_number, available_quantity
FROM items WHERE deleted_at IS NULL
"""

# ONLY vendor source available in this module's table subset — see module
# docstring for why organisation_item_mappings cannot be used here.
_ITEM_VENDORS = """
SELECT id, item_id, vendor_id, vendor_name, lead_time_days, unit_price
FROM item_vendors
"""

_PURCHASE_ORDERS = """
SELECT id, reference_no, vendor_id, organisation_id, warehouse_id,
       scheduled_delivery_date, status, sent_at, created_at
FROM purchase_orders
"""

_PURCHASE_ORDER_LINES = """
SELECT id, purchase_order_id, item_id, item_name, ordered_quantity,
       received_quantity, unit_price
FROM purchase_order_lines
"""

# status/updated_at feed the grn_on_time() status-transition proxy — never a
# true "received date" column (none exists anywhere in the schema).
_GRNS = """
SELECT id, reference_no, purchase_order_id, vendor_id, warehouse_id,
       scheduled_delivery_date, status, is_partially_closed, created_at, updated_at
FROM goods_received_notes
"""


def read_delay_tables(tenant: str = "demo") -> tuple[dict[str, pd.DataFrame], object]:
    """Bulk read of every raw table M3 needs, operator PII pseudonymized in-place."""
    da = get_data_access()
    md = load_md_map(da, tenant)
    salt = get_settings().hmac_salt.get_secret_value()

    work_orders = da.query(_WORK_ORDERS, tenant=tenant)
    work_orders = pseudonymize_operators(work_orders, ["assigned_operators"], salt)

    time_logs = da.query(_WORK_ORDER_TIME_LOGS, tenant=tenant)
    time_logs = pseudonymize_operators(time_logs, ["operator_id"], salt)

    tables = {
        "manufacturing_orders": da.query(_MANUFACTURING_ORDERS, tenant=tenant),
        "operations": da.query(_OPERATIONS, tenant=tenant),
        "operation_dependencies": da.query(_OPERATION_DEPENDENCIES, tenant=tenant),
        "work_centers": da.query(_WORK_CENTERS, tenant=tenant),
        "work_orders": work_orders,
        "work_order_time_logs": time_logs,
        "mo_components": da.query(_MO_COMPONENTS, tenant=tenant),
        "items": da.query(_ITEMS, tenant=tenant),
        "item_vendors": da.query(_ITEM_VENDORS, tenant=tenant),
        "purchase_orders": da.query(_PURCHASE_ORDERS, tenant=tenant),
        "purchase_order_lines": da.query(_PURCHASE_ORDER_LINES, tenant=tenant),
        "goods_received_notes": da.query(_GRNS, tenant=tenant),
    }
    return tables, md
