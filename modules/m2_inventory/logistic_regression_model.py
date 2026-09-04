"""Explainable logistic-regression weekly hazard model for M2 inventory."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from m2_inventory.hazard_model_base import BaseInventoryHazardModel


class LogisticRegressionInventoryModel(BaseInventoryHazardModel):
    algorithm_name = "logistic-regression"
    scale_numeric_features = True

    def build_estimator(self):
        return LogisticRegression(
            C=1.0,
            max_iter=2_000,
            random_state=self.random_state,
            solver="liblinear",
        )


if __name__ == "__main__":
    from m2_inventory.csv_training import run_model_cli

    run_model_cli(LogisticRegressionInventoryModel)
