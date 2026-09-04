"""LightGBM weekly hazard model for M2 inventory."""

from __future__ import annotations

from lightgbm import LGBMClassifier

from m2_inventory.hazard_model_base import BaseInventoryHazardModel


class LightGBMInventoryModel(BaseInventoryHazardModel):
    algorithm_name = "lightgbm"

    def build_estimator(self):
        return LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )


if __name__ == "__main__":
    from m2_inventory.csv_training import run_model_cli

    run_model_cli(LightGBMInventoryModel)
