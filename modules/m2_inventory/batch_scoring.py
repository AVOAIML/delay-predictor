"""Weekly M2 batch scoring using the published inventory hazard-model contract."""

from __future__ import annotations

import json

import pandas as pd

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger
from maxxflow_core.jsonutil import json_default
from maxxflow_mlops.registry import MLflowRegistry
from m2_inventory.csv_training import MODULE, _coerce_model_input, require_hazard_artifact
from m2_inventory.db_prediction import read_db_prediction_snapshots

log = get_logger("m2_inventory.batch_scoring")

HAZARD_OUTPUT_COLUMNS = (
    "model_risk_30d",
    "model_risk_60d",
    "horizon_calibrated",
    "projection_based",
    "risk_30d_difference",
    "risk_60d_difference",
    "business_rule_flags",
    "business_rule_override",
    "final_alert",
    "alert_reasons",
    "risk_30d",
    "risk_60d",
    "badge_30d",
    "badge_60d",
    "is_rule_based",
    "suppressed",
    "model_version",
)


def score_records(model, records: pd.DataFrame) -> pd.DataFrame:
    """Score raw inventory snapshots through an MLflow hazard pyfunc."""
    predictions = model.predict(_coerce_model_input(records))
    missing = sorted(set(HAZARD_OUTPUT_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(
            "Published M2 model does not implement the inventory hazard contract; "
            f"missing outputs: {missing}"
        )
    return predictions


def _payload(row: pd.Series, registry_version: str | None) -> dict:
    values = {column: row[column] for column in HAZARD_OUTPUT_COLUMNS}
    values["artifact_model_version"] = values["model_version"]
    values["model_version"] = registry_version
    values["scored_at"] = get_clock().as_of().isoformat()
    return {"ai_stockout": values}


def score(tenant: str = "demo") -> int:
    """Score current DB snapshots and cache the rich hazard result on each item."""
    from maxxflow_data.engine import get_data_access

    snapshots = read_db_prediction_snapshots(tenant, get_clock().as_of())
    if snapshots.empty:
        return 0

    registry = MLflowRegistry()
    resolved = registry.resolve_champion_name(tenant=tenant, module=MODULE)
    if resolved is None:
        raise RuntimeError(
            f"No published M2 champion exists for tenant {tenant!r} or the global base model"
        )
    serving_name, _ = resolved
    version = registry.get_alias_version(name=serving_name, alias="champion")
    require_hazard_artifact(registry, serving_name, str(version))
    model = registry.load_champion(name=serving_name)
    predictions = score_records(model, snapshots)

    data_access = get_data_access()
    for position, (_, row) in enumerate(predictions.iterrows()):
        item_id = snapshots.iloc[position]["item_id"]
        data_access.execute(
            "UPDATE items SET custom_elements = COALESCE(custom_elements,'{}'::jsonb) "
            "|| CAST(:p AS jsonb) WHERE id = :id",
            {
                "p": json.dumps(_payload(row, version), default=json_default),
                "id": item_id,
            },
            tenant=tenant,
        )
        if bool(row["suppressed"]):
            data_access.execute(
                "INSERT INTO audit_logs (module, action, entity_type, entity_id, metadata, timestamp) "
                "VALUES ('Predictive Inventory Alerts','suppress','Item',:id,CAST(:m AS jsonb),now())",
                {
                    "id": item_id,
                    "m": json.dumps(
                        {
                            "reason": "open PO covers deficit AND vendor on-time>95%",
                            "model_risk_30d": row["model_risk_30d"],
                        },
                        default=json_default,
                    ),
                },
                tenant=tenant,
            )

    log.info("M2 hazard batch scored %d items (model v%s)", len(predictions), version)
    return len(predictions)
