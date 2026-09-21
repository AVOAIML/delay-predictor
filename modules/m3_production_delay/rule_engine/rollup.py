"""Assembles the job/operations/operators/components rollup JSON from the flat
tables `dal.read_delay_tables()` returns. Pure transform — no DB access here.

Mirrors exactly the AVAILABLE / PARTIAL / NOT AVAILABLE split documented in
`dal.py`'s docstring. Nothing here fabricates a value that isn't backed by a
real column or a documented approximation:

  operation_id            -> work_orders.id                                  (real)
  depends_on_operation_ids -> operation_dependencies, translated from the BOM
                              Operations level to THIS job's WorkOrder ids     (real)
  sequence_no             -> ALWAYS None — bom_operation_orders is not part
                              of this module's local schema subset           (NOT AVAILABLE)
  operation_type / status -> resolved via the inverted MasterDataMap         (real)
  expected/actual_duration, job_quantity, current_done_quantity
                           -> work_orders columns directly                    (real)
  operator_count           -> len(assigned_operators)                        (derived)
  operator.name            -> ALWAYS None here — assigned_operators only
                              carries the HMAC-pseudonymized token by the
                              time it reaches this module (PII never leaves
                              the DAL); resolving a display name requires a
                              separate, access-controlled lookup outside this
                              pipeline, never a plain users.name join here    (NOT AVAILABLE)
  last_10_work_orders.elapsed_time_minutes
                           -> SUM(work_order_time_logs.duration_minutes) for
                              that operator token on that work order (NOT
                              work_orders.real_duration, which is the total
                              across every assigned operator)                 (real)
  component required/available_quantity -> mo_components / items             (real)
  vendor                   -> item_vendors only (this module's only vendor
                              source; organisation_item_mappings is not
                              part of the local schema subset)                (real, legacy source)
  po_order_date             -> purchase_orders.sent_at, falling back to
                              created_at                                     (PARTIAL/approximated)
  po_order_deadline         -> purchase_orders.scheduled_delivery_date       (real)
  grn_received_date         -> goods_received_notes.updated_at, gated on
                              status in {'Goods Received','Partially
                              Received'} — the same status-transition proxy
                              `transforms.grn_on_time` already uses for M2.
                              NOT a true received-date column (none exists).  (PARTIAL/approximated)
"""

from __future__ import annotations

import pandas as pd

from maxxflow_data.masterdata_map import MasterDataMap

_RECEIVED_STATUSES = {"Goods Received", "Partially Received"}


def _invert_md(md: MasterDataMap) -> dict[str, str]:
    """uuid -> code. MasterDataMap only exposes the forward (code -> uuid) direction."""
    return {v: k[1] for k, v in md.code_to_id.items()}


def _code_for(id_to_code: dict[str, str], master_data_id: str | None) -> str | None:
    if master_data_id is None:
        return None
    return id_to_code.get(master_data_id)


def _index_work_orders_by_operator(work_orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One-time O(n) index: operator token -> the work orders they were assigned to.

    `assigned_operators` has no reverse index in the database, so *something*
    has to scan `work_orders` to answer "which WOs was this operator on" — but
    that scan only needs to happen ONCE per batch, not once per operator per
    job. `explode()` turns the one row-per-WO / array-of-operators table into
    one row per (WO, operator) pair; grouping that by operator token then
    gives an O(1) lookup for every operator, however many jobs/operations
    are being rolled up together.
    """
    if work_orders.empty:
        return {}
    exploded = work_orders.explode("assigned_operators")
    exploded = exploded[exploded["assigned_operators"].notna()]
    return {token: rows for token, rows in exploded.groupby("assigned_operators")}


def _last_n_work_orders_for_operator(
    operator_token: str,
    operator_index: dict[str, pd.DataFrame],
    time_logs: pd.DataFrame,
    operations: pd.DataFrame,
    id_to_code: dict[str, str],
    n: int = 10,
) -> list[dict]:
    """Every WO this operator was assigned to, most recently completed first.

    Reads from the pre-built `operator_index` (see `_index_work_orders_by_operator`)
    instead of re-scanning the full `work_orders` table — O(1) per operator
    however many jobs/operators are being processed in the same batch.
    """
    wos = operator_index.get(operator_token)
    if wos is None or wos.empty:
        return []
    wos = wos.sort_values("actual_end", ascending=False, na_position="last").head(n)

    ops_by_id = operations.set_index("id") if not operations.empty else operations

    history = []
    for _, wo in wos.iterrows():
        own_logs = time_logs[
            (time_logs["work_order_id"] == wo["id"])
            & (time_logs["operator_id"] == operator_token)
        ]
        elapsed = int(own_logs["duration_minutes"].fillna(0).sum())

        op_type_code = None
        if not ops_by_id.empty and wo["operation_id"] in ops_by_id.index:
            op_type_code = _code_for(
                id_to_code, ops_by_id.loc[wo["operation_id"], "operation_type_id"]
            )

        history.append({
            "work_order_id": wo["id"],
            "elapsed_time_minutes": elapsed,
            "scheduled_time_minutes": int(wo["expected_duration"]),
            "completed_on": wo["actual_end"],
            "operation_type": op_type_code,
        })
    return history


def _vendor_for_item(
    item_id: str,
    item_vendors: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    purchase_order_lines: pd.DataFrame,
    grns: pd.DataFrame,
    n: int = 10,
) -> dict | None:
    vendor_rows = item_vendors[item_vendors["item_id"] == item_id]
    if vendor_rows.empty:
        return None
    vendor_row = vendor_rows.iloc[0]  # no is_primary flag available at this layer — first match
    vendor_id = vendor_row["vendor_id"]

    lines = purchase_order_lines[purchase_order_lines["item_id"] == item_id]
    pos = purchase_orders[
        purchase_orders["id"].isin(lines["purchase_order_id"])
        & (purchase_orders["vendor_id"] == vendor_id)
    ].copy()
    pos = pos.sort_values("created_at", ascending=False).head(n)

    purchase_orders_out = []
    for _, po in pos.iterrows():
        po_order_date = po["sent_at"] if pd.notna(po["sent_at"]) else po["created_at"]

        grn_rows = grns[grns["purchase_order_id"] == po["id"]]
        grn_received_date = None
        received = grn_rows[grn_rows["status"].isin(_RECEIVED_STATUSES)]
        if not received.empty:
            grn_received_date = received.iloc[0]["updated_at"]  # proxy — see module docstring

        purchase_orders_out.append({
            "po_number": po["reference_no"],
            "po_order_date": po_order_date,
            "po_order_deadline": po["scheduled_delivery_date"],
            "grn_received_date": grn_received_date,
        })

    return {
        "vendor_id": vendor_id,
        "name": vendor_row["vendor_name"],
        "last_10_purchase_orders": purchase_orders_out,
    }


def _predecessor_work_order_ids(
    wo_bom_operation_id: str,
    job_wo_ids_by_bom_operation: dict[str, list[str]],
    depends_on_by_bom_operation: dict[str, list[str]],
) -> list[str]:
    """The `operation_id` (WorkOrder.id, this rollup's own field of that name)
    of every predecessor of `wo_bom_operation_id` WITHIN THIS SAME JOB.

    `operation_dependencies` is defined at the BOM's Operations level
    (`operations.id`), not per-WorkOrder — a dependency is "this BOM step
    depends on that BOM step", not tied to any one job. So resolving it to
    "which of THIS job's work orders is the predecessor" needs one more hop:
    BOM operation the predecessor is defined on -> the WorkOrder this job
    created for that same BOM operation.
    """
    predecessor_bom_ids = depends_on_by_bom_operation.get(wo_bom_operation_id, [])
    result: list[str] = []
    for bom_id in predecessor_bom_ids:
        result.extend(job_wo_ids_by_bom_operation.get(bom_id, []))
    return result


def _build_operations_for_mo(
    mo_id: str,
    tables: dict[str, pd.DataFrame],
    id_to_code: dict[str, str],
    operator_index: dict[str, pd.DataFrame],
    depends_on_by_bom_operation: dict[str, list[str]],
) -> list[dict]:
    """The `operations` array for ONE manufacturing order. Shared by both
    `build_job_rollup` (single job) and `build_job_rollups` (batch) so the
    per-operation/operator/component logic exists in exactly one place."""
    work_orders = tables["work_orders"]
    time_logs = tables["work_order_time_logs"]
    operations = tables["operations"]
    mo_components = tables["mo_components"]
    items = tables["items"]
    item_vendors = tables["item_vendors"]
    purchase_orders = tables["purchase_orders"]
    purchase_order_lines = tables["purchase_order_lines"]
    grns = tables["goods_received_notes"]

    job_work_orders = work_orders[work_orders["mo_id"] == mo_id]
    job_components = mo_components[mo_components["mo_id"] == mo_id]

    # BOM operation id -> this job's WorkOrder id(s) for that operation — the
    # hop `_predecessor_work_order_ids` needs to translate a BOM-level
    # dependency edge into "which of THIS job's operations is that".
    job_wo_ids_by_bom_operation: dict[str, list[str]] = {}
    for _, wo in job_work_orders.iterrows():
        job_wo_ids_by_bom_operation.setdefault(wo["operation_id"], []).append(wo["id"])

    operations_out = []
    for _, wo in job_work_orders.iterrows():
        op_row = operations[operations["id"] == wo["operation_id"]]
        op_type_code = (
            _code_for(id_to_code, op_row.iloc[0]["operation_type_id"])
            if not op_row.empty else None
        )
        status_code = _code_for(id_to_code, wo["status_id"])

        assigned = wo["assigned_operators"] or []
        operators_out = []
        for operator_token in assigned:
            operators_out.append({
                "operator_id": operator_token,  # pseudonymized token, NOT the raw uuid
                "name": None,  # NOT AVAILABLE here — see module docstring
                "last_10_work_orders": _last_n_work_orders_for_operator(
                    operator_token, operator_index, time_logs, operations, id_to_code,
                ),
            })

        components_out = []
        for _, comp in job_components.iterrows():
            item_row = items[items["id"] == comp["item_id"]]
            if item_row.empty:
                continue
            item_row = item_row.iloc[0]
            components_out.append({
                "component_id": comp["item_id"],
                "name": item_row["item_name"],
                "required_quantity": float(comp["required_qty"]),
                "available_quantity": float(item_row["available_quantity"]),
                "vendor": _vendor_for_item(
                    comp["item_id"], item_vendors, purchase_orders,
                    purchase_order_lines, grns,
                ),
            })

        operations_out.append({
            "operation_id": wo["id"],
            "operation_type": op_type_code,
            "status": status_code,
            "depends_on_operation_ids": _predecessor_work_order_ids(
                wo["operation_id"], job_wo_ids_by_bom_operation, depends_on_by_bom_operation,
            ),
            "expected_duration_minutes": int(wo["expected_duration"]),
            "actual_duration_minutes": (
                int(wo["real_duration"]) if pd.notna(wo["real_duration"]) else None
            ),
            "job_quantity": float(wo["quantity"]),
            "current_done_quantity": float(wo["units_done"]),
            "operator_count": len(assigned),
            "operators": operators_out,
            "components": components_out,
        })

    return operations_out


def build_job_rollups(
    tables: dict[str, pd.DataFrame],
    md: MasterDataMap,
    job_references: list[str] | None = None,
) -> list[dict]:
    """Assembles rollups for one or more jobs in one pass.

    `job_references=None` rolls up every manufacturing order present in
    `tables["manufacturing_orders"]`; pass an explicit list (even a
    single-item one, e.g. `["WH/MO/00142"]`) to scope it to specific jobs —
    there is no separate single-job function, this covers both.

    The operator index (`_index_work_orders_by_operator`) and the inverted
    MasterData map are each built ONCE here and reused across every job in
    the batch — the win grows with how much operator overlap there is
    across jobs.
    """
    id_to_code = _invert_md(md)
    operator_index = _index_work_orders_by_operator(tables["work_orders"])

    depends_on_by_bom_operation: dict[str, list[str]] = {}
    op_deps = tables["operation_dependencies"]
    if not op_deps.empty:
        for operation_id, rows in op_deps.groupby("operation_id"):
            depends_on_by_bom_operation[operation_id] = rows["depends_on_id"].tolist()

    mos = tables["manufacturing_orders"]
    if job_references is not None:
        mos = mos[mos["reference"].isin(job_references)]

    rollups = []
    for _, mo in mos.iterrows():
        operations_out = _build_operations_for_mo(
            mo["id"], tables, id_to_code, operator_index, depends_on_by_bom_operation,
        )
        rollups.append({"job_id": mo["reference"], "operations": operations_out})
    return rollups
