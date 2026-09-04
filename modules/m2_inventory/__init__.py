"""M2 — calibrated weekly inventory hazard models and batch alerts.

The authoritative feature contract lives in :mod:`m2_inventory.inventory_dataset`.
M2 compares four hazard estimators during training and publishes only their common
``InventoryHazardPyfunc`` contract. It is scored as a weekly batch, not through the
online multi-model endpoint.
"""

from m2_inventory.inventory_dataset import (
    FEATURE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    InventoryDatasetBuilder,
)
from m2_inventory.pipeline import drift, fe, score, score_from_csv, score_from_mlflow, train

__all__ = [
    "FEATURE_COLUMNS",
    "MODEL_INPUT_COLUMNS",
    "InventoryDatasetBuilder",
    "fe",
    "train",
    "score",
    "score_from_csv",
    "score_from_mlflow",
    "drift",
]
