"""Random-Forest weekly hazard model for M2 inventory."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from m2_inventory.hazard_model_base import BaseInventoryHazardModel


class RandomForestInventoryModel(BaseInventoryHazardModel):
    algorithm_name = "random-forest"

    def build_estimator(self):
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=self.random_state,
        )


if __name__ == "__main__":
    from m2_inventory.csv_training import run_model_cli

    run_model_cli(RandomForestInventoryModel)
