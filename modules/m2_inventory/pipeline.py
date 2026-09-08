"""Canonical M2 CLI hooks for hazard training, batch scoring, and drift."""

from __future__ import annotations

import os

from maxxflow_features.lake import get_lake
from maxxflow_mlops.drift import get_drift_reporter
from m2_inventory.inventory_dataset import (
    DEFAULT_DATASET_PATH,
    NUMERIC_FEATURES,
    InventoryDatasetBuilder,
)

MODULE = "m2_inventory"


def fe(tenant: str = "demo") -> str:
    """Persist an engineered snapshot of the hazard-model DB prediction input."""
    from maxxflow_core.clock import get_clock
    from m2_inventory.db_prediction import read_db_prediction_snapshots

    snapshots = read_db_prediction_snapshots(tenant, get_clock().as_of())
    feats = InventoryDatasetBuilder().engineer_features(snapshots)
    return get_lake().write_parquet(feats, "gold", "features", tenant=tenant, module=MODULE)


def train(
    tenant: str = "demo",
    *,
    dataset_path=DEFAULT_DATASET_PATH,
    artifact_dir="artifacts/m2_inventory",
    seed: int = 42,
    register: bool = True,
):
    """Train and compare all four hazard candidates using chronological splits."""
    from m2_inventory.csv_training import train_all_csv_models

    summary = train_all_csv_models(
        dataset_path,
        artifact_dir,
        seed,
        tenant=tenant,
        register=register,
    )
    return summary.get("version") if register else summary


def score_from_csv(
                   tenant: str = "demo",
                   dataset_path=DEFAULT_DATASET_PATH,
                   artifact_dir="artifacts/m2_inventory"):
    """Score each latest item/warehouse snapshot with the CSV-model winner."""
    from m2_inventory.csv_training import score_latest_csv_snapshots

    return score_latest_csv_snapshots(tenant, dataset_path, artifact_dir)


def score_from_mlflow(tenant: str = "demo",
                      dataset_path=DEFAULT_DATASET_PATH):
    """Score latest snapshots through the tenant/global MLflow champion."""
    from m2_inventory.csv_training import score_latest_mlflow_snapshots

    return score_latest_mlflow_snapshots(tenant, dataset_path)


def score(tenant: str = "demo") -> int:
    from m2_inventory.batch_scoring import score as _score

    return _score(tenant)


def drift(tenant: str = "demo") -> str:
    from maxxflow_core.clock import get_clock
    from m2_inventory.db_prediction import read_db_prediction_snapshots

    lake = get_lake()
    try:
        reference = lake.read_parquet("gold", "features", tenant=tenant, module=MODULE)
    except FileNotFoundError:
        return f"no baseline yet for tenant={tenant!r} — run `make fe MODULE={MODULE}` first"
    snapshots = read_db_prediction_snapshots(tenant, get_clock().as_of())
    current = InventoryDatasetBuilder().engineer_features(snapshots)
    out_dir = os.environ.get("DRIFT_OUT", "reports")
    return get_drift_reporter().report(reference[NUMERIC_FEATURES], current[NUMERIC_FEATURES],
                                       name=f"{tenant}_{MODULE}", out_dir=out_dir)["report_path"]
