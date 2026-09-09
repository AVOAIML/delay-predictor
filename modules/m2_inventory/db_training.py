"""PostgreSQL snapshot-history adapter for M2 training."""

from __future__ import annotations

import pandas as pd

from maxxflow_core.settings import get_settings
from maxxflow_data.engine import get_data_access
from m2_inventory.inventory_dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES

TABLE = "inventory_ml_snapshots"
REQUIRED_COLUMNS = {
    "snapshot_id",
    "item_id",
    "warehouse_id",
    "snapshot_date",
    "stockout_flag",
}
NUMERIC_ZERO_COLUMNS = set(NUMERIC_FEATURES) | {
    "months_of_history",
    "external_risk_pct",
    "planned_bom_qty",
    "past_due_qty",
}
CATEGORICAL_DEFAULT_COLUMNS = set(CATEGORICAL_FEATURES) - {"warehouse_id"}
BOOLEAN_DEFAULT_COLUMNS = {"use_rule_based"}
DATE_DEFAULT_COLUMNS = {"open_po_deadline"}
BOOTSTRAP_WEEKS = 80
BOOTSTRAP_ITEM_LIMIT = 50
MIN_HISTORY_WEEKS = 80


def _columns(tenant: str) -> tuple[str | None, set[str]]:
    da = get_data_access()
    schema_result = da.query("SELECT current_schema() AS schema", tenant=tenant)
    schema = str(schema_result["schema"].iloc[0]) if not schema_result.empty else None
    result = da.query(
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = current_schema() AND table_name = '{TABLE}'",
        tenant=tenant,
    )
    return schema, {str(value) for value in result.get("column_name", [])}


def apply_column_defaults(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    out = frame.copy()
    defaulted: list[dict] = []
    for column in sorted(NUMERIC_ZERO_COLUMNS - set(out.columns)):
        out[column] = 0.0
        defaulted.append({"column": column, "default": 0})
    for column in sorted(CATEGORICAL_DEFAULT_COLUMNS - set(out.columns)):
        out[column] = "unknown"
        defaulted.append({"column": column, "default": "unknown"})
    for column in sorted(BOOLEAN_DEFAULT_COLUMNS - set(out.columns)):
        out[column] = False
        defaulted.append({"column": column, "default": False})
    for column in sorted(DATE_DEFAULT_COLUMNS - set(out.columns)):
        out[column] = pd.NaT
        defaulted.append({"column": column, "default": None})
    return out, defaulted


def _history_problem(columns: set[str], frame: pd.DataFrame | None = None) -> str | None:
    if not columns:
        return f"{TABLE} does not exist"
    missing = sorted(REQUIRED_COLUMNS - columns)
    if missing:
        return f"{TABLE} is missing required columns: {missing}"
    if frame is None:
        return None
    if frame.empty:
        return f"{TABLE} returned no rows"
    dates = pd.to_datetime(frame["snapshot_date"], errors="coerce").nunique()
    classes = pd.to_numeric(frame["stockout_flag"], errors="coerce").nunique()
    if dates < MIN_HISTORY_WEEKS or classes < 2:
        return (
            f"{TABLE} needs at least {MIN_HISTORY_WEEKS} weeks and both stockout classes"
        )
    return None


def _bootstrap_training_frame(tenant: str, reason: str) -> pd.DataFrame:
    from m2_inventory.db_prediction import read_db_prediction_snapshots

    current = read_db_prediction_snapshots(tenant, pd.Timestamp.now(tz="UTC"))
    if current.empty:
        raise ValueError(
            f"{reason}; operational items also returned no rows, so training cannot start"
        )
    current, _ = apply_column_defaults(current.head(BOOTSTRAP_ITEM_LIMIT))
    start = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    start -= pd.Timedelta(weeks=BOOTSTRAP_WEEKS - 1)
    rows: list[dict] = []
    for item_index, record in enumerate(current.to_dict(orient="records")):
        base_available = pd.to_numeric(record.get("available_qty"), errors="coerce")
        rop = pd.to_numeric(record.get("rop"), errors="coerce")
        base_available = max(0.0 if pd.isna(base_available) else float(base_available), 1.0)
        rop = max(0.0 if pd.isna(rop) else float(rop), 1.0)
        for scenario in ("stable", "risk", "delayed"):
            for week in range(BOOTSTRAP_WEEKS):
                row = dict(record)
                cycle = 5 if scenario == "risk" else 12
                phase = week % cycle
                event_phase = 4 if scenario == "delayed" else cycle - 1
                stockout = scenario != "stable" and phase == event_phase
                if scenario == "stable":
                    available = max(base_available, rop * 2.0 + 1.0)
                elif stockout:
                    available = 0.0
                elif phase == (event_phase - 1) % cycle:
                    available = max(0.25, rop * 0.25)
                else:
                    available = max(1.0, min(base_available, rop))
                row.update({
                    "snapshot_id": f"bootstrap-{item_index}-{scenario}-{week}",
                    "item_id": f"{record['item_id']}::bootstrap-{scenario}",
                    "snapshot_date": start + pd.Timedelta(weeks=week),
                    "available_qty": available,
                    "stockout_flag": stockout,
                    "months_of_history": 12.0,
                    "use_rule_based": False,
                })
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs.update({
        "fallback_used": True,
        "fallback_reason": reason,
        "source_name": "MaXXflow Database (bootstrap fallback)",
    })
    return frame


def build_training_frame(tenant: str) -> pd.DataFrame:
    schema, columns = _columns(tenant)
    expected_schema = get_settings().tenant_schema(tenant)
    if schema != expected_schema:
        raise ValueError(f"tenant schema {expected_schema!r} does not exist")
    problem = _history_problem(columns)
    if problem is None:
        try:
            frame = get_data_access().query(
                f"SELECT * FROM {TABLE} ORDER BY snapshot_date, item_id, warehouse_id",
                tenant=tenant,
            )
            problem = _history_problem(columns, frame)
            if problem is None:
                return apply_column_defaults(frame)[0]
        except Exception as exc:
            problem = f"{TABLE} could not be read ({type(exc).__name__}: {exc})"
    return _bootstrap_training_frame(tenant, problem)


def describe_sources(tenant: str) -> dict:
    schema, columns = _columns(tenant)
    expected_schema = get_settings().tenant_schema(tenant)
    present = bool(columns)
    rows = None
    snapshot_dates = 0
    target_classes = 0
    history_error = None
    if present:
        da = get_data_access()
        try:
            if REQUIRED_COLUMNS <= columns:
                stats = da.query(
                    f"SELECT count(*) AS n, count(DISTINCT snapshot_date) AS snapshot_dates, "
                    f"count(DISTINCT stockout_flag) AS target_classes FROM {TABLE}",
                    tenant=tenant,
                ).iloc[0]
                rows = int(stats["n"])
                snapshot_dates = int(stats["snapshot_dates"])
                target_classes = int(stats["target_classes"])
            else:
                rows = int(da.query(
                    f"SELECT count(*) AS n FROM {TABLE}", tenant=tenant
                )["n"].iloc[0])
        except Exception as exc:
            history_error = f"{type(exc).__name__}: {exc}"
    missing = sorted(REQUIRED_COLUMNS - columns)
    defaulted = sorted(
        (NUMERIC_ZERO_COLUMNS | CATEGORICAL_DEFAULT_COLUMNS |
         BOOLEAN_DEFAULT_COLUMNS | DATE_DEFAULT_COLUMNS) - columns
    ) if present else []
    history_ready = bool(
        present and rows and not missing
        and snapshot_dates >= MIN_HISTORY_WEEKS and target_classes >= 2
    )
    fallback_rows = 0
    fallback_error = None
    if not history_ready:
        try:
            from m2_inventory.db_prediction import read_db_prediction_snapshots
            fallback_rows = len(
                read_db_prediction_snapshots(tenant, pd.Timestamp.now(tz="UTC"))
            )
        except Exception as exc:
            fallback_error = f"{type(exc).__name__}: {exc}"
    return {
        "connected": True,
        "schema": schema,
        "expected_schema": expected_schema,
        "schema_exists": schema == expected_schema,
        "sources": [{
            "table": TABLE,
            "present": present,
            "rows": rows,
            "snapshot_dates": snapshot_dates,
            "target_classes": target_classes,
            "required_columns": sorted(REQUIRED_COLUMNS),
            "missing_columns": missing,
            "defaulted_columns": defaulted,
            "error": history_error,
        }, {
            "table": "operational inventory tables",
            "present": fallback_rows > 0,
            "rows": fallback_rows,
            "bootstrap_fallback": True,
            "error": fallback_error,
        }],
        "passthrough_columns": sorted(columns),
        "derived_features": sorted(NUMERIC_FEATURES),
        "trainable": history_ready or fallback_rows > 0,
        "fallback_used": not history_ready and fallback_rows > 0,
        "fallback_warning": (
            "Training will use generated bootstrap history and its metrics are not "
            "historical performance evidence."
            if not history_ready and fallback_rows > 0 else None
        ),
    }
