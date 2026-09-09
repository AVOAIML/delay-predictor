"""Reusable training, calibration and survival logic for all M2 classifiers."""

from __future__ import annotations

import threading
import warnings
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from mlflow.pyfunc import PythonModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from m2_inventory.business_rules import (
    BADGE_RED_THRESHOLD,
    MODEL_ALERT_THRESHOLD,
    RULE_BASED_HISTORY_MONTHS,
    SUPPRESSION_RELIABILITY_THRESHOLD,
    InventoryBusinessRuleEvaluator,
)
from m2_inventory.feature_diagnostics import (
    audit_numeric_features,
    enforce_training_safeguards,
    high_numeric_correlations,
    serving_ood_report,
)
from m2_inventory.inventory_dataset import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    InventoryDatasetBuilder,
    TemporalDatasetSplit,
)


def expected_calibration_error(y_true, probability, bins: int = 10) -> float:
    """Return equal-width expected calibration error."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def calibration_table(y_true, probability, bins: int = 10) -> list[dict[str, float | int]]:
    """Return an equal-width reliability table suitable for reports and curves."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    table = []
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            table.append({
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "rows": int(mask.sum()),
                "mean_probability": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            })
    return table


class ProbabilityCalibrator:
    """Isotonic calibration with a Platt-scaling fallback for sparse data."""

    def __init__(self):
        self.method: str | None = None
        self.model: Any = None

    def fit(self, raw_probability, target) -> "ProbabilityCalibrator":
        raw = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
        y = np.asarray(target, dtype=int)
        class_counts = np.bincount(y, minlength=2)
        if len(y) >= 500 and class_counts.min() >= 30 and np.unique(raw).size >= 20:
            self.method = "isotonic"
            self.model = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
        else:
            self.method = "platt"
            self.model = LogisticRegression(C=1e6, solver="lbfgs").fit(raw.reshape(-1, 1), y)
        return self

    def predict(self, raw_probability) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator has not been fitted")
        raw = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            calibrated = self.model.predict(raw)
        else:
            calibrated = self.model.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, 0.0, 1.0)


class ConstrainedHorizonCalibrator:
    """Calibrate nested cumulative horizons without breaking their ordering.

    The long-horizon event is decomposed into the short-horizon event plus the
    conditional probability of an event in the remaining interval.  Both
    components are fitted on calibration rows only.
    """

    def __init__(self):
        self.short = ProbabilityCalibrator()
        self.increment = ProbabilityCalibrator()

    @property
    def method(self) -> str:
        return f"short_{self.short.method}+conditional_increment_{self.increment.method}"

    def fit(
        self,
        raw_short,
        short_target,
        raw_short_long_eligible,
        raw_long,
        short_target_long_eligible,
        long_target,
    ) -> "ConstrainedHorizonCalibrator":
        self.short.fit(raw_short, short_target)
        short_y = np.asarray(short_target_long_eligible, dtype=int)
        long_y = np.asarray(long_target, dtype=int)
        at_risk = short_y == 0
        if not at_risk.any():
            raise ValueError("No post-30-day at-risk rows remain for horizon calibration")
        raw_increment = self._conditional_increment(
            np.asarray(raw_short_long_eligible, dtype=float)[at_risk],
            np.asarray(raw_long, dtype=float)[at_risk],
        )
        self.increment.fit(raw_increment, long_y[at_risk])
        return self

    def predict(self, raw_short, raw_long) -> tuple[np.ndarray, np.ndarray]:
        calibrated_short = self.short.predict(raw_short)
        raw_increment = self._conditional_increment(raw_short, raw_long)
        calibrated_increment = self.increment.predict(raw_increment)
        calibrated_long = calibrated_short + (
            (1.0 - calibrated_short) * calibrated_increment
        )
        return (
            np.clip(calibrated_short, 0.0, 1.0),
            np.clip(calibrated_long, 0.0, 1.0),
        )

    @staticmethod
    def _conditional_increment(raw_short, raw_long) -> np.ndarray:
        short = np.clip(np.asarray(raw_short, dtype=float), 0.0, 1.0)
        long = np.clip(np.asarray(raw_long, dtype=float), short, 1.0)
        denominator = np.maximum(1.0 - short, 1e-12)
        return np.clip((long - short) / denominator, 0.0, 1.0)


class BaseInventoryHazardModel(ABC):
    """Template method shared by Logistic, RF, LightGBM and XGBoost."""

    algorithm_name = "base"
    # A common standardized space makes SMOTE's nearest-neighbour distances
    # comparable across numeric features and across all four algorithms.
    scale_numeric_features = True

    def __init__(
        self,
        random_state: int = 42,
        *,
        feature_columns: list[str] | tuple[str, ...] | None = None,
        numeric_features: list[str] | tuple[str, ...] | None = None,
        categorical_features: list[str] | tuple[str, ...] | None = None,
    ):
        self.random_state = random_state
        # Persist the fitted feature contract in each artifact.  Older artifacts
        # do not have these attributes and continue to fall back to the current
        # production constants through the helper methods below.
        self.feature_columns_ = list(feature_columns or FEATURE_COLUMNS)
        self.numeric_features_ = list(numeric_features or NUMERIC_FEATURES)
        self.categorical_features_ = list(
            categorical_features or CATEGORICAL_FEATURES
        )
        if set(self.numeric_features_) | set(self.categorical_features_) != set(
            self.feature_columns_
        ):
            raise ValueError(
                "numeric and categorical feature contracts must exactly cover "
                "feature_columns"
            )
        self.dataset_builder = InventoryDatasetBuilder()
        self.preprocessor = self._build_preprocessor()
        self.estimator = self.build_estimator()
        self.calibrator = ProbabilityCalibrator()
        # Independent calibrators are retained as a diagnostic benchmark only.
        self.horizon_calibrators: dict[int, ProbabilityCalibrator] = {}
        self.constrained_horizon_calibrator: ConstrainedHorizonCalibrator | None = None
        self.horizon_baseline_probability_: dict[int, float] = {}
        self.model_version = f"{date.today().isoformat()}-{self.algorithm_name}-v1"
        self.training_distribution_: dict[str, dict[int, int]] = {}
        self.feature_audit_: pd.DataFrame | None = None
        self.feature_correlation_audit_: pd.DataFrame | None = None
        self.feature_safeguard_warnings_: list[str] = []
        self.last_ood_report_: pd.DataFrame | None = None
        self._warned_ood_features: set[str] = set()
        # Guards the read-check-update of _warned_ood_features below. Today's
        # batch scoring calls predict_weekly_hazard sequentially so this never
        # contends, but the model instance is cached/shared (see
        # maxxflow_mlops.serving.ModelRouter's LRU cache) — a future concurrent
        # serving path hitting the same instance from multiple threads would
        # otherwise race on this shared mutable set (lost updates, duplicate
        # warnings). Cheap to hold even uncontended, so it's in from the start.
        self._ood_warn_lock = threading.Lock()
        self.is_fitted = False

    def __getstate__(self) -> dict[str, Any]:
        """Exclude the process-local lock from persisted model state."""
        state = self.__dict__.copy()
        state.pop("_ood_warn_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore model state and create a fresh lock in the loading process."""
        self.__dict__.update(state)
        self._ood_warn_lock = threading.Lock()

    @abstractmethod
    def build_estimator(self):
        """Create the algorithm-specific binary classifier."""

    def fit(self, split: TemporalDatasetSplit) -> "BaseInventoryHazardModel":
        self.feature_audit_ = audit_numeric_features(
            split.train, split.test, self._numeric_features()
        )
        self.feature_correlation_audit_ = high_numeric_correlations(
            split.train, self._numeric_features()
        )
        self.feature_safeguard_warnings_ = enforce_training_safeguards(
            self.feature_audit_
        )
        train_x = split.train[self._feature_columns()]
        transformed_train = np.asarray(self.preprocessor.fit_transform(train_x))
        training_target = split.train["target"].astype(int).to_numpy()
        self.training_distribution_["original"] = self._class_counts(training_target)

        # Resampling is deliberately confined to training data. Calibration and
        # test retain their natural class rates so probability calibration and
        # reported metrics remain representative of production.
        smote = SMOTE(random_state=self.random_state)
        min_required = smote.k_neighbors + 1
        class_counts = self.training_distribution_["original"]
        if len(class_counts) < 2 or min(class_counts.values()) < min_required:
            raise ValueError(
                f"{self.algorithm_name}: training split has too few stockout "
                f"examples to oversample — class counts {class_counts}, but SMOTE "
                f"needs at least {min_required} examples of each class "
                f"(k_neighbors={smote.k_neighbors} + 1). Widen the training date "
                "range or check upstream label generation before retrying."
            )
        smote_x, smote_y = smote.fit_resample(transformed_train, training_target)
        self.training_distribution_["after_smote"] = self._class_counts(smote_y)
        tomek = TomekLinks(sampling_strategy="all")
        resampled_x, resampled_y = tomek.fit_resample(smote_x, smote_y)
        self.training_distribution_["after_tomek"] = self._class_counts(resampled_y)

        self.estimator.fit(resampled_x, resampled_y)
        weekly_calibration = self._calibration_before_test(split, horizon_weeks=1)
        raw_calibration = self._raw_probability(
            np.asarray(
                self.preprocessor.transform(
                    weekly_calibration[self._feature_columns()]
                )
            )
        )
        self.calibrator.fit(raw_calibration, weekly_calibration["target"].astype(int))
        self.is_fitted = True

        # The weekly calibrator corrects conditional one-week hazards. Survival
        # aggregation can still be miscalibrated at 30/60 days, so calibrate those
        # final horizons separately. Origins whose outcome windows reach the test
        # period are purged to keep the future test genuinely untouched.
        calibration_risks = self._predict_survival_risks(
            split.calibration, max_weeks=9, apply_horizon_calibration=False
        )
        horizon_training: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
        for horizon, risk in ((4, calibration_risks[0]), (9, calibration_risks[1])):
            target_column = f"target_{horizon}w"
            if target_column not in split.calibration:
                continue
            calibration = self._calibration_before_test(split, horizon_weeks=horizon)
            valid = calibration[target_column].notna()
            calibration = calibration.loc[valid]
            positions = split.calibration.index.get_indexer(calibration.index)
            target = calibration[target_column].astype(int).to_numpy()
            calibrator = ProbabilityCalibrator().fit(risk[positions], target)
            self.horizon_calibrators[horizon] = calibrator
            self.horizon_baseline_probability_[horizon] = float(target.mean())
            horizon_training[horizon] = (calibration, positions, target)

        if 4 in horizon_training and 9 in horizon_training:
            calibration_4w, positions_4w, target_4w = horizon_training[4]
            calibration_9w, positions_9w, target_9w = horizon_training[9]
            target_4w_for_9w = calibration_9w["target_4w"].astype(int).to_numpy()
            self.constrained_horizon_calibrator = ConstrainedHorizonCalibrator().fit(
                calibration_risks[0][positions_4w],
                target_4w,
                calibration_risks[0][positions_9w],
                calibration_risks[1][positions_9w],
                target_4w_for_9w,
                target_9w,
            )
        return self

    def predict_weekly_hazard(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        engineered = self.dataset_builder.engineer_features(frame)
        feature_audit = getattr(self, "feature_audit_", None)
        if feature_audit is not None:
            self.last_ood_report_ = serving_ood_report(
                engineered, feature_audit
            )
            observed_features = set(self.last_ood_report_.get("feature", []))
            # Lazily create+store the lock for artifacts pickled before this
            # attribute existed (see the "Older artifacts" note in __init__);
            # every model built via __init__ already has one, so this branch
            # is dead for anything trained after this change.
            lock = getattr(self, "_ood_warn_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._ood_warn_lock = lock
            with lock:
                warned_features = getattr(self, "_warned_ood_features", set())
                new_features = observed_features - warned_features
                if new_features:
                    warnings.warn(
                        "M2 serving OOD/support warning; extreme |z| or synthetic-only "
                        "training support for features: "
                        f"{sorted(new_features)}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    warned_features.update(new_features)
                    self._warned_ood_features = warned_features
        raw = self._raw_probability(
            self.preprocessor.transform(engineered[self._feature_columns()])
        )
        return self.calibrator.predict(raw)

    def predict_risk(self, frame: pd.DataFrame, max_weeks: int = 9) -> pd.DataFrame:
        """Return pure model risks plus a separate deterministic alert decision."""
        if max_weeks < 9:
            raise ValueError("max_weeks must be at least 9 to calculate 60-day risk")
        origins = self.dataset_builder.engineer_features(frame).reset_index(drop=True)
        risk_30d, risk_60d = self._predict_survival_risks(
            origins, max_weeks=max_weeks, apply_horizon_calibration=True
        )
        projection_risk_30d, projection_risk_60d = (
            self._predict_projection_based_risks(origins, max_weeks=max_weeks)
        )

        out = pd.DataFrame(index=origins.index)
        for column in ("item_id", "warehouse_id"):
            if column in origins:
                out[column] = origins[column].to_numpy()
        out["model_risk_30d"] = np.clip(risk_30d, 0.0, 1.0)
        out["model_risk_60d"] = np.clip(risk_60d, 0.0, 1.0)
        out["horizon_calibrated"] = [
            {"risk_30d": float(short), "risk_60d": float(long)}
            for short, long in zip(risk_30d, risk_60d)
        ]
        out["projection_based"] = [
            {"risk_30d": float(short), "risk_60d": float(long)}
            for short, long in zip(projection_risk_30d, projection_risk_60d)
        ]
        out["risk_30d_difference"] = risk_30d - projection_risk_30d
        out["risk_60d_difference"] = risk_60d - projection_risk_60d
        # Backward-compatible aliases remain model probabilities; deterministic
        # policy never overwrites them.
        out["risk_30d"] = out["model_risk_30d"]
        out["risk_60d"] = out["model_risk_60d"]
        out["badge_30d"] = out["model_risk_30d"].map(self._badge)
        out["badge_60d"] = out["model_risk_60d"].map(self._badge)
        out["is_rule_based"] = self._rule_based_mask(origins)

        rule_results = InventoryBusinessRuleEvaluator.evaluate(origins)
        model_alert = out["model_risk_30d"].ge(MODEL_ALERT_THRESHOLD)
        out["business_rule_override"] = [result.override for result in rule_results]
        out["business_rule_flags"] = [result.flags for result in rule_results]
        out["final_alert"] = model_alert.to_numpy() | out[
            "business_rule_override"
        ].to_numpy()
        out["alert_reasons"] = [
            (["model_risk_30d"] if bool(model_alert.iloc[position]) else [])
            + result.reasons
            for position, result in enumerate(rule_results)
        ]

        net_available = origins["available_qty"].fillna(0.0) - origins["reserved_qty"].fillna(0.0)
        deficit = (origins["rop"].fillna(0.0) - net_available).clip(lower=0.0)
        reliability = self._numeric_column(
            origins, "open_po_vendor_reliability", np.nan
        ).fillna(self._numeric_column(origins, "calculated_vendor_reliability", np.nan))
        # Unknown reliability defaults to 0.0 — never enough to clear
        # SUPPRESSION_RELIABILITY_THRESHOLD below — rather than some neutral or
        # optimistic value. Suppressing an alert is an affirmative claim that
        # the open PO will actually cover the deficit, and a vendor with no
        # track record hasn't earned that claim. This is deliberately the same
        # conservative direction as business_rules.UNRELIABLE_PO_COVERAGE,
        # which also treats missing reliability as unreliable rather than
        # assuming it's fine — so the two don't drift apart again.
        reliability = reliability.fillna(
            self._numeric_column(origins, "computed_reliability", 0.0)
        ).fillna(0.0)
        out["suppressed"] = (
            (deficit > 0.0)
            & (origins["open_qty"].fillna(0.0) >= deficit)
            & (reliability > SUPPRESSION_RELIABILITY_THRESHOLD)
        )
        out["model_version"] = self.model_version
        return out

    def _predict_survival_risks(
        self,
        frame: pd.DataFrame,
        *,
        max_weeks: int,
        apply_horizon_calibration: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate pure calibrated weekly model hazards without policy overrides."""
        origins = self.dataset_builder.engineer_features(frame).reset_index(drop=True)
        weekly = []
        for week_index in range(max_weeks):
            projected = self.dataset_builder.project_week(origins, week_index)
            hazard = self.predict_weekly_hazard(projected)
            weekly.append(hazard)
        hazards = np.column_stack(weekly)
        risk_30d = 1.0 - np.prod(1.0 - hazards[:, :4], axis=1)
        risk_60d = 1.0 - np.prod(1.0 - hazards[:, :9], axis=1)
        if apply_horizon_calibration:
            constrained = getattr(self, "constrained_horizon_calibrator", None)
            if constrained is not None:
                risk_30d, risk_60d = constrained.predict(risk_30d, risk_60d)
            else:
                # Compatibility for artifacts trained before constrained
                # horizon calibration was introduced.
                calibrators = getattr(self, "horizon_calibrators", {})
                if 4 in calibrators:
                    risk_30d = calibrators[4].predict(risk_30d)
                if 9 in calibrators:
                    risk_60d = calibrators[9].predict(risk_60d)
        return np.clip(risk_30d, 0.0, 1.0), np.clip(risk_60d, 0.0, 1.0)

    def _predict_projection_based_risks(
        self, frame: pd.DataFrame, *, max_weeks: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score richer weekly projections without final horizon calibration."""
        origins = self.dataset_builder.engineer_features(frame).reset_index(drop=True)
        weekly = []
        for week_index in range(max_weeks):
            projected = self.dataset_builder.project_week_projection_based(
                origins, week_index
            )
            weekly.append(self.predict_weekly_hazard(projected))
        hazards = np.column_stack(weekly)
        risk_30d = 1.0 - np.prod(1.0 - hazards[:, :4], axis=1)
        risk_60d = 1.0 - np.prod(1.0 - hazards[:, :9], axis=1)
        return np.clip(risk_30d, 0.0, 1.0), np.clip(risk_60d, 0.0, 1.0)

    @staticmethod
    def _calibration_before_test(
        split: TemporalDatasetSplit, horizon_weeks: int
    ) -> pd.DataFrame:
        test_start = split.test["snapshot_date"].min()
        outcome_end = split.calibration["snapshot_date"] + pd.to_timedelta(
            7 * horizon_weeks, unit="D"
        )
        calibration = split.calibration.loc[outcome_end.lt(test_start)].copy()
        if calibration.empty:
            raise ValueError(
                f"No calibration rows remain before the test boundary for {horizon_weeks} weeks"
            )
        return calibration

    def evaluate(self, test: pd.DataFrame) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        metrics.update(self._evaluate_weekly_metrics(test))
        metrics.update(self._evaluate_horizon_metrics(test))
        metrics.update(self._evaluate_projection_based_metrics(test))
        return metrics

    def _evaluate_weekly_metrics(self, test: pd.DataFrame) -> dict[str, Any]:
        """Identity fields plus one-week-ahead hazard metrics."""
        weekly = self.predict_weekly_hazard(test)
        y_week = test["target"].astype(int).to_numpy()
        weekly_class = weekly >= 0.5
        positive_rate = float(y_week.mean())
        accuracy = float((weekly_class == y_week).mean())
        base_rate_accuracy = float(max(positive_rate, 1.0 - positive_rate))
        metrics: dict[str, Any] = {
            "algorithm": self.algorithm_name,
            "calibration_method": str(self.calibrator.method),
            "test_rows": int(len(test)),
            "weekly_auc": float(roc_auc_score(y_week, weekly)),
            "weekly_average_precision": float(average_precision_score(y_week, weekly)),
            "weekly_brier": float(brier_score_loss(y_week, weekly)),
            "weekly_ece": expected_calibration_error(y_week, weekly),
            "accuracy": accuracy,
            "base_rate_accuracy": base_rate_accuracy,
            "accuracy_over_base_rate": accuracy - base_rate_accuracy,
            "positive_rate": positive_rate,
        }
        for stage, counts in self.training_distribution_.items():
            metrics[f"{stage}_negative_rows"] = counts.get(0, 0)
            metrics[f"{stage}_positive_rows"] = counts.get(1, 0)
        return metrics

    def _evaluate_horizon_metrics(self, test: pd.DataFrame) -> dict[str, Any]:
        """30/60-day calibrated-risk metrics: order-violation checks, per-horizon
        AUC/Brier/ECE against the production and independently-calibrated
        paths, and reliability diagrams."""
        metrics: dict[str, Any] = {}
        risks = self.predict_risk(test)
        horizon_gap = risks["model_risk_30d"] - risks["model_risk_60d"]
        horizon_order_violation = horizon_gap.gt(1e-12)
        metrics["risk_horizon_order_violations"] = int(horizon_order_violation.sum())
        metrics["risk_horizon_order_violation_rate"] = float(
            horizon_order_violation.mean()
        )
        metrics["risk_horizon_order_max_gap"] = float(
            horizon_gap.clip(lower=0.0).max()
        )
        survival_risk = self._predict_survival_risks(
            test, max_weeks=9, apply_horizon_calibration=False
        )
        independent_risk = self._independently_calibrated_horizon_risks(*survival_risk)
        independent_gap = independent_risk[0] - independent_risk[1]
        metrics["independent_risk_horizon_order_violations"] = int(
            (independent_gap > 1e-12).sum()
        )
        metrics["independent_risk_horizon_order_max_gap"] = float(
            np.clip(independent_gap, 0.0, None).max()
        )
        for horizon, risk_column, before_probability in (
            (4, "model_risk_30d", survival_risk[0]),
            (9, "model_risk_60d", survival_risk[1]),
        ):
            label_column = f"target_{horizon}w"
            mask = test[label_column].notna().to_numpy()
            if mask.any() and test.loc[mask, label_column].nunique() > 1:
                y_horizon = test.loc[mask, label_column].astype(int)
                probability = risks.loc[mask, risk_column]
                before = before_probability[mask]
                prefix = "risk_30d" if horizon == 4 else "risk_60d"
                baseline_probability = getattr(
                    self, "horizon_baseline_probability_", {}
                ).get(horizon, float(y_horizon.mean()))
                baseline = np.full(len(y_horizon), baseline_probability)
                metrics[f"{prefix}_test_rows"] = int(mask.sum())
                metrics[f"{prefix}_auc"] = float(roc_auc_score(y_horizon, probability))
                metrics[f"{prefix}_average_precision"] = float(
                    average_precision_score(y_horizon, probability)
                )
                metrics[f"{prefix}_brier"] = float(brier_score_loss(y_horizon, probability))
                metrics[f"{prefix}_ece"] = expected_calibration_error(y_horizon, probability)
                metrics[f"{prefix}_brier_before_horizon_calibration"] = float(
                    brier_score_loss(y_horizon, before)
                )
                metrics[f"{prefix}_ece_before_horizon_calibration"] = (
                    expected_calibration_error(y_horizon, before)
                )
                independent_probability = independent_risk[0 if horizon == 4 else 1][mask]
                metrics[f"{prefix}_independent_brier"] = float(
                    brier_score_loss(y_horizon, independent_probability)
                )
                metrics[f"{prefix}_independent_ece"] = expected_calibration_error(
                    y_horizon, independent_probability
                )
                metrics[f"{prefix}_calibration_table_independent"] = calibration_table(
                    y_horizon, independent_probability
                )
                metrics[f"{prefix}_baseline_probability"] = float(baseline_probability)
                metrics[f"{prefix}_baseline_brier"] = float(
                    brier_score_loss(y_horizon, baseline)
                )
                metrics[f"{prefix}_brier_improvement_vs_baseline"] = (
                    metrics[f"{prefix}_baseline_brier"] - metrics[f"{prefix}_brier"]
                )
                calibrator = getattr(self, "horizon_calibrators", {}).get(horizon)
                metrics[f"{prefix}_horizon_calibration_method"] = (
                    getattr(self.constrained_horizon_calibrator, "method", None)
                )
                metrics[f"{prefix}_calibration_table_before"] = calibration_table(
                    y_horizon, before
                )
                metrics[f"{prefix}_calibration_table_after"] = calibration_table(
                    y_horizon, probability
                )
        return metrics

    def _evaluate_projection_based_metrics(self, test: pd.DataFrame) -> dict[str, Any]:
        """Diagnostic-only metrics for the richer projection-based risk path,
        scored independently of the production calibrated-risk path above."""
        metrics: dict[str, Any] = {}
        projection_risks = self._predict_projection_based_risks(test, max_weeks=9)
        projection_gap = projection_risks[0] - projection_risks[1]
        metrics["projection_based_risk_horizon_order_violations"] = int(
            (projection_gap > 1e-12).sum()
        )
        metrics["projection_based_risk_horizon_order_max_gap"] = float(
            np.clip(projection_gap, 0.0, None).max()
        )
        for horizon, probability in ((4, projection_risks[0]), (9, projection_risks[1])):
            label_column = f"target_{horizon}w"
            mask = test[label_column].notna().to_numpy()
            if mask.any() and test.loc[mask, label_column].nunique() > 1:
                y_horizon = test.loc[mask, label_column].astype(int)
                projected_probability = probability[mask]
                prefix = "projection_based_risk_30d" if horizon == 4 else "projection_based_risk_60d"
                metrics[f"{prefix}_auc"] = float(
                    roc_auc_score(y_horizon, projected_probability)
                )
                metrics[f"{prefix}_average_precision"] = float(
                    average_precision_score(y_horizon, projected_probability)
                )
                metrics[f"{prefix}_brier"] = float(
                    brier_score_loss(y_horizon, projected_probability)
                )
                metrics[f"{prefix}_ece"] = expected_calibration_error(
                    y_horizon, projected_probability
                )
                metrics[f"risk_{'30d' if horizon == 4 else '60d'}_calibration_table_projection_based"] = calibration_table(
                    y_horizon, projected_probability
                )
        return metrics

    def _independently_calibrated_horizon_risks(
        self, raw_30d: np.ndarray, raw_60d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the previous independent approach for evaluation only."""
        calibrators = getattr(self, "horizon_calibrators", {})
        risk_30d = (
            calibrators[4].predict(raw_30d) if 4 in calibrators else raw_30d
        )
        risk_60d = (
            calibrators[9].predict(raw_60d) if 9 in calibrators else raw_60d
        )
        return np.clip(risk_30d, 0.0, 1.0), np.clip(risk_60d, 0.0, 1.0)

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "BaseInventoryHazardModel":
        model = joblib.load(path)
        if not isinstance(model, BaseInventoryHazardModel):
            raise TypeError(f"{path} does not contain an inventory hazard model")
        return model

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
        ]
        if self.scale_numeric_features:
            numeric_steps.append(("scale", StandardScaler()))
        numeric = Pipeline(numeric_steps)
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            [
                ("numeric", numeric, self._numeric_features()),
                ("categorical", categorical, self._categorical_features()),
            ],
            remainder="drop",
        )

    def _feature_columns(self) -> list[str]:
        return list(getattr(self, "feature_columns_", FEATURE_COLUMNS))

    def _numeric_features(self) -> list[str]:
        return list(getattr(self, "numeric_features_", NUMERIC_FEATURES))

    def _categorical_features(self) -> list[str]:
        return list(getattr(self, "categorical_features_", CATEGORICAL_FEATURES))

    def _raw_probability(self, transformed) -> np.ndarray:
        # Some macOS Accelerate/Numpy combinations emit a spurious "overflow
        # encountered in matmul"/"...in dot" RuntimeWarning from a finite
        # matrix multiplication. Silence only that message — not RuntimeWarning
        # category-wide — so an unrelated, possibly-real RuntimeWarning from
        # any of the four algorithms still surfaces. np.isfinite below is the
        # actual correctness check either way.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            probability = self.estimator.predict_proba(np.asarray(transformed))
        positive = np.asarray(probability[:, 1], dtype=float)
        if not np.isfinite(positive).all():
            raise ValueError(f"{self.algorithm_name} produced non-finite probabilities")
        return positive

    @staticmethod
    def _rule_based_mask(frame: pd.DataFrame) -> np.ndarray:
        # Missing history means "unknown", not automatically less than the
        # cutoff. Explicit exports can still opt into the deterministic fallback.
        history = BaseInventoryHazardModel._numeric_column(
            frame, "months_of_history", RULE_BASED_HISTORY_MONTHS
        ).fillna(RULE_BASED_HISTORY_MONTHS)
        explicit = frame.get("use_rule_based", pd.Series(False, index=frame.index))
        explicit = explicit.astype("boolean").fillna(False).astype(bool)
        return ((history < RULE_BASED_HISTORY_MONTHS) | explicit).to_numpy()

    @staticmethod
    def _rule_based_hazard(frame: pd.DataFrame) -> np.ndarray:
        available = frame["available_qty"].fillna(0.0) - frame["reserved_qty"].fillna(0.0)
        demand = frame["demand_forecast_qty"].fillna(0.0) + frame["past_due_qty"].fillna(0.0)
        demand += BaseInventoryHazardModel._numeric_column(
            frame, "planned_bom_qty", 0.0
        ).fillna(0.0)
        supply = available.clip(lower=0.0) + frame["open_qty"].fillna(0.0)
        shortage = (demand - supply).clip(lower=0.0)
        denominator = np.maximum(demand.to_numpy(), frame["rop"].fillna(0.0).to_numpy())
        return np.clip(shortage.to_numpy() / np.maximum(denominator, 1.0), 0.0, 1.0)

    @staticmethod
    def _badge(probability: float) -> str:
        if probability >= BADGE_RED_THRESHOLD:
            return "Red"
        if probability >= MODEL_ALERT_THRESHOLD:
            return "Amber"
        return "Green"

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("model has not been fitted")

    @staticmethod
    def _class_counts(target) -> dict[int, int]:
        values, counts = np.unique(np.asarray(target, dtype=int), return_counts=True)
        return {int(value): int(count) for value, count in zip(values, counts)}

    @staticmethod
    def _numeric_column(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
        values = frame[name] if name in frame else pd.Series(default, index=frame.index)
        return pd.to_numeric(values, errors="coerce")


class InventoryHazardPyfunc(PythonModel):
    """MLflow serving adapter around a reusable inventory hazard model."""

    def __init__(self, hazard_model: BaseInventoryHazardModel):
        self.hazard_model = hazard_model
        self.algorithm_name = hazard_model.algorithm_name
        self.model_version = hazard_model.model_version

    def predict(
        self,
        context,
        model_input: pd.DataFrame,
        params=None,
    ) -> pd.DataFrame:
        frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        return self.hazard_model.predict_risk(frame)
