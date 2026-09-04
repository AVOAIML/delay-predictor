"""XGBoost weekly hazard model for M2 inventory."""

from __future__ import annotations

from m2_inventory.hazard_model_base import BaseInventoryHazardModel


class XGBoostInventoryModel(BaseInventoryHazardModel):
    algorithm_name = "xgboost"

    def build_estimator(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
            raise ImportError(
                "XGBoostInventoryModel requires the project's 'xgboost' dependency"
            ) from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=350,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
        )


if __name__ == "__main__":
    from m2_inventory.csv_training import run_model_cli

    run_model_cli(XGBoostInventoryModel)
