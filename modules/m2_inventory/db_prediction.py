"""Build current M2 hazard-model snapshots from tenant operational tables.

This adapter is prediction-only. The operational schema contains current state,
but not the weekly history and future outcomes required to retrain the discrete-
time hazard model. It therefore never presents current DB rows as training data.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import numpy as np
import pandas as pd

from maxxflow_core.errors import get_logger
from m2_inventory.business_rules import RULE_BASED_HISTORY_MONTHS
from m2_inventory.inventory_dataset import MODEL_INPUT_COLUMNS

log = get_logger("m2_inventory.db_prediction")

_MO_STATUS_LABELS = {
    "DRAFT": "Draft",
    "CONFIRMED": "Confirmed",
    "IN_PROGRESS": "In Progress",
    "DONE": "Done",
    "CANCELLED": "Cancelled",
}
_CLOSED_PO_STATUSES = {
    "closed", "cancelled", "canceled", "goods received", "received",
    "partially received - closed",
}


def _frame(tables: Mapping[str, object], name: str) -> pd.DataFrame:
    value = tables.get(name)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _numeric(value, default: float = 0.0) -> float:
    result = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(result) else float(result)


def _custom_value(custom, name: str, default=None):
    if not isinstance(custom, Mapping):
        return default
    if name in custom:
        return custom[name]
    for namespace in ("inventory", "m2_inventory", "ai_inventory"):
        nested = custom.get(namespace)
        if isinstance(nested, Mapping) and name in nested:
            return nested[name]
    return default


def _mo_status_lookup(md) -> dict[str, str]:
    values = getattr(md, "code_to_id", {}) or {}
    return {
        str(identifier): _MO_STATUS_LABELS.get(str(code), str(code).replace("_", " ").title())
        for (category, code), identifier in values.items()
        if str(category) == "MO_STATUS"
    }


def _vendor_delivery_history(grns: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """Return completed/on-time/late counts and earliest observation per vendor."""
    if grns.empty:
        return {}, {}, {}, {}
    data = grns.copy()
    status = data.get("status", pd.Series("", index=data.index)).astype("string").str.casefold()
    data = data[status.str.contains("received", na=False)].copy()
    data["_deadline"] = pd.to_datetime(data.get("scheduled_delivery_date"), errors="coerce")
    data["_received"] = pd.to_datetime(data.get("updated_at"), errors="coerce")
    data = data.dropna(subset=["vendor_id", "_deadline", "_received"])
    if data.empty:
        return {}, {}, {}, {}

    # A partially received PO may have several GRNs. Count each completed PO once,
    # using its final receipt timestamp against its scheduled deadline.
    keys = ["vendor_id", "purchase_order_id"] if "purchase_order_id" in data else ["vendor_id", "id"]
    deliveries = data.groupby(keys, dropna=False).agg(
        deadline=("_deadline", "max"), received=("_received", "max")
    ).reset_index()
    deliveries["on_time"] = deliveries["received"] <= deliveries["deadline"]
    grouped = deliveries.groupby("vendor_id")
    total = grouped.size().astype(int).to_dict()
    on_time = grouped["on_time"].sum().astype(int).to_dict()
    late = {vendor: total[vendor] - on_time.get(vendor, 0) for vendor in total}
    earliest = grouped["received"].min().to_dict()
    return total, on_time, late, earliest


def build_db_prediction_snapshots(
    tables: Mapping[str, object],
    md,
    as_of: datetime | pd.Timestamp,
) -> pd.DataFrame:
    """Create one current model-input row per item and default warehouse."""
    items = _frame(tables, "items")
    if items.empty:
        return pd.DataFrame(columns=MODEL_INPUT_COLUMNS)

    warehouses = _frame(tables, "warehouses")
    warehouse_by_id = warehouses.set_index("id") if not warehouses.empty else pd.DataFrame()
    vendors = _frame(tables, "item_vendors")
    vendor_by_item = {
        item_id: group.iloc[0]
        for item_id, group in vendors.groupby("item_id", sort=False)
    } if not vendors.empty else {}

    grns = _frame(tables, "goods_received_notes")
    vendor_total, vendor_on_time, vendor_late, vendor_earliest = _vendor_delivery_history(grns)
    vendor_reliability = {
        vendor: vendor_on_time.get(vendor, 0) / total
        for vendor, total in vendor_total.items() if total > 0
    }

    orders = _frame(tables, "manufacturing_orders")
    components = _frame(tables, "mo_components")
    status_lookup = _mo_status_lookup(md)
    if not orders.empty:
        orders = orders.copy()
        orders["_status"] = orders["status_id"].astype(str).map(status_lookup).fillna("Not Applicable")
    component_orders = (
        components.merge(orders[["id", "_status", "warehouse_id"]], left_on="mo_id", right_on="id",
                         how="left", suffixes=("", "_mo"))
        if not components.empty and not orders.empty else pd.DataFrame()
    )
    component_orders_by_item = (
        {item_id: group for item_id, group in component_orders.groupby("item_id", sort=False)}
        if not component_orders.empty else {}
    )

    pos = _frame(tables, "purchase_orders")
    po_lines = _frame(tables, "purchase_order_lines")
    open_lines = pd.DataFrame()
    if not pos.empty and not po_lines.empty:
        active = ~pos.get("status", pd.Series("", index=pos.index)).astype("string").str.casefold().isin(
            _CLOSED_PO_STATUSES
        )
        active_pos = pos[active].copy()
        open_lines = po_lines.merge(
            active_pos[["id", "vendor_id", "warehouse_id", "scheduled_delivery_date"]],
            left_on="purchase_order_id", right_on="id", how="inner", suffixes=("", "_po"),
        )
        open_lines["_open_qty"] = (
            pd.to_numeric(open_lines["ordered_quantity"], errors="coerce").fillna(0.0)
            - pd.to_numeric(open_lines["received_quantity"], errors="coerce").fillna(0.0)
        ).clip(lower=0.0)
        open_lines = open_lines[open_lines["_open_qty"] > 0].copy()
    open_lines_by_item = (
        {item_id: group for item_id, group in open_lines.groupby("item_id", sort=False)}
        if not open_lines.empty else {}
    )

    boms = _frame(tables, "boms")
    bom_components = _frame(tables, "bom_components")
    products_using: dict = {}
    if not boms.empty and not bom_components.empty:
        usage = bom_components.dropna(subset=["item_id"]).merge(
            boms[["id", "product_id"]], left_on="bom_id", right_on="id", how="left",
            suffixes=("", "_bom"),
        )
        products_using = usage.groupby("item_id")["product_id_bom"].nunique().to_dict()

    as_of_ts = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of)
    rows: list[dict] = []
    skipped_item_ids: list = []
    for item in items.to_dict(orient="records"):
        item_id = item["id"]
        warehouse_id = item.get("default_warehouse_id")
        if warehouse_id is None or pd.isna(warehouse_id):
            skipped_item_ids.append(item_id)
            continue
        warehouse = (
            warehouse_by_id.loc[warehouse_id]
            if not warehouse_by_id.empty and warehouse_id in warehouse_by_id.index else {}
        )
        custom = item.get("custom_elements")
        primary = vendor_by_item.get(item_id)
        primary_vendor = primary.get("vendor_id") if primary is not None else None

        mo = component_orders_by_item.get(item_id, pd.DataFrame())
        draft = mo[mo["_status"].eq("Draft")] if not mo.empty else pd.DataFrame()
        active = mo[mo["_status"].isin(["Confirmed", "In Progress"])] if not mo.empty else pd.DataFrame()
        active_statuses = set(active["_status"]) if not active.empty else set()
        mo_status = "In Progress" if "In Progress" in active_statuses else (
            "Confirmed" if "Confirmed" in active_statuses else "Not Applicable"
        )

        incoming = open_lines_by_item.get(item_id, pd.DataFrame())
        if not incoming.empty and "warehouse_id" in incoming:
            at_warehouse = incoming[ incoming["warehouse_id"].isna() | incoming["warehouse_id"].eq(warehouse_id) ]
            if not at_warehouse.empty:
                incoming = at_warehouse
        open_qty = float(incoming["_open_qty"].sum()) if not incoming.empty else 0.0
        deadline = pd.to_datetime(incoming["scheduled_delivery_date"], errors="coerce").min() \
            if not incoming.empty else pd.NaT
        if not incoming.empty:
            rel = incoming["vendor_id"].map(vendor_reliability)
            known = rel.notna()
            open_vendor_rel = float(np.average(rel[known], weights=incoming.loc[known, "_open_qty"])) \
                if known.any() else np.nan
        else:
            open_vendor_rel = np.nan

        months = 0.0
        earliest = vendor_earliest.get(primary_vendor)
        if earliest is not None and not pd.isna(earliest):
            months = max(0.0, (as_of_ts - pd.Timestamp(earliest)).days / 30.0)
        forecast = _numeric(item.get("forecasted_quantity"))
        demand = _custom_value(custom, "demand_forecast_qty", forecast)

        rows.append({
            "item_id": str(item_id),
            "item_name": item.get("item_name"),
            "part_number": item.get("part_number"),
            "warehouse_id": str(warehouse_id),
            "warehouse_name": warehouse.get("warehouse_name"),
            "snapshot_date": as_of_ts.normalize(),
            "item_type": item.get("item_type"),
            "unit_cost": item.get("unit_cost"),
            "unit_of_measurement": item.get("unit_of_measurement"),
            "warehouse_type": warehouse.get("warehouse_type"),
            "available_qty": _numeric(item.get("available_quantity")),
            "reserved_qty": _numeric(active.get("reserved_qty", pd.Series(dtype=float)).sum()),
            "forecasted_qty": forecast,
            "rop": _numeric(item.get("rop")),
            "n_products_using_item": _numeric(products_using.get(item_id)),
            "demand_forecast_qty": _numeric(demand),
            "past_due_qty": _numeric(_custom_value(custom, "past_due_qty", 0.0)),
            "open_qty": open_qty,
            "open_po_deadline": deadline,
            "lead_time_days": _numeric(primary.get("lead_time_days") if primary is not None else 0.0),
            "vendor_total_completed_pos": _numeric(vendor_total.get(primary_vendor)),
            "vendor_on_time_pos": _numeric(vendor_on_time.get(primary_vendor)),
            "vendor_late_pos": _numeric(vendor_late.get(primary_vendor)),
            "calculated_vendor_reliability": vendor_reliability.get(primary_vendor, np.nan),
            "open_po_vendor_reliability": open_vendor_rel,
            "draft_mo_status": "Draft" if not draft.empty else "Not Applicable",
            "draft_mo_required_qty": _numeric(draft.get("required_qty", pd.Series(dtype=float)).sum()),
            "mo_status": mo_status,
            "consumed_qty": _numeric(active.get("consumed_qty", pd.Series(dtype=float)).sum()),
            "months_of_history": months,
            "use_rule_based": months < RULE_BASED_HISTORY_MONTHS,
            "external_risk_pct": _numeric(_custom_value(custom, "external_risk_pct", 0.0)),
        })

    if skipped_item_ids:
        shown = skipped_item_ids[:20]
        more = f" (+{len(skipped_item_ids) - len(shown)} more)" if len(skipped_item_ids) > len(shown) else ""
        log.warning(
            "M2 prediction snapshot: skipped %d/%d items with no default warehouse "
            "set — never scored for stockout risk this run: %s%s",
            len(skipped_item_ids), len(items), shown, more,
        )

    return pd.DataFrame(rows)


def read_db_prediction_snapshots(tenant: str, as_of: datetime | pd.Timestamp) -> pd.DataFrame:
    """Read tenant tables through the shared DAL and build current snapshots."""
    from m2_inventory.dal import read_inventory_tables

    tables, md = read_inventory_tables(tenant)
    return build_db_prediction_snapshots(tables, md, as_of)
