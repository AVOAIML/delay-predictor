"""CSV preparation for the M2 discrete-time inventory hazard models.

At each surviving item/warehouse snapshot in week ``t``, the model estimates
the conditional probability of ``stockout_flag`` in week ``t + 1``. Only the
explicitly approved operational features below are exposed to estimators; the
current event, risk flags and validation columns remain audit-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASET_PATH = Path(
    "dataset/M2data/all_verticals_business_scenarios_v3.csv"
)
KEY_COLUMNS = ["item_id", "warehouse_id"]
EVENT_COLUMN = "stockout_flag"
DEMAND_FORECAST_WEEKS = 4.0
_BASE_LEAD_TIME_COLUMN = "_base_lead_time_days"

# Legacy generated-training schema retained for compatibility checks. These are
# columns produced by enrichment/scenario generation, not required source data.
BUSINESS_SCENARIO_REQUIRED_COLUMNS = {
    "draft_mo_status",
    "draft_mo_required_qty",
    "draft_mo_shortage_qty",
    "draft_mo_component_status",
    "calculated_has_draft_mo_stock_risk",
    "mo_status",
    "consumed_qty",
    "material_overrun_qty",
    "has_material_overrun",
    "unreserved_available_qty",
    "material_shortfall_qty",
    "calculated_has_material_overrun_risk",
    "vendor_total_completed_pos",
    "vendor_on_time_pos",
    "vendor_late_pos",
    "calculated_vendor_reliability",
    "calculated_open_po_vendor_below_75pct",
    "projected_without_po",
    "projected_with_po",
    "po_needed_for_coverage",
    "unreliable_po_coverage",
    "calculated_low_stock_risk",
    "calculated_stockout_risk_alert",
    "risk_reason_count",
    "risk_reasons",
    "business_scenario_injected",
    "synthetic_business_scenario",
    "business_scenario_role",
    "business_scenario_positive",
    "business_scenario_holdout",
    "future_outcome_date_1w",
    "future_stockout_flag_1w",
}

# These columns belong to ML training/audit data, never to the client/Prisma
# source contract. Scenario generators may populate them; ordinary operational
# snapshots receive neutral defaults and causal future labels inside ``load``.
ML_AUDIT_COLUMN_DEFAULTS = {
    "business_scenario_injected": False,
    "synthetic_business_scenario": "NONE",
    "business_scenario_role": "NONE",
    "business_scenario_positive": False,
    "business_scenario_holdout": False,
    "scenario_type": "NONE",
}
ML_AUDIT_COLUMNS = frozenset(
    {
        *ML_AUDIT_COLUMN_DEFAULTS,
        "future_outcome_date_1w",
        "future_stockout_flag_1w",
    }
)

# Minimum historical source contract. Every field is an observed operational
# fact. In particular, no target-at-a-future-horizon or synthetic audit marker
# is accepted as a required client/Prisma field.
OPERATIONAL_REQUIRED_COLUMNS = frozenset(
    {
        "snapshot_id",
        "item_id",
        "warehouse_id",
        "snapshot_date",
        "available_qty",
        "reserved_qty",
        "rop",
        "demand_forecast_qty",
        "open_po_vendor_reliability",
        EVENT_COLUMN,
    }
)

EXCLUDED_LEAKAGE_COLUMNS = [
    EVENT_COLUMN,
    "stockout_risk_alert",
    "has_draft_mo_stock_risk",
    "has_material_overrun_risk",
    "open_po_vendor_below_75pct",
    "calculated_low_stock_risk",
    "calculated_has_draft_mo_stock_risk",
    "calculated_has_material_overrun_risk",
    "calculated_open_po_vendor_below_75pct",
    "unreliable_po_coverage",
    "calculated_stockout_risk_alert",
    "risk_reason_count",
    "risk_reasons",
    "draft_mo_flag_matches",
    "material_overrun_flag_matches",
    "vendor_flag_matches",
    "stockout_alert_matches",
    "has_material_overrun",
    "primary_vendor_below_75pct",
    "business_scenario_injected",
    "synthetic_business_scenario",
    "business_scenario_role",
    "business_scenario_positive",
    "business_scenario_holdout",
    "future_outcome_date_1w",
    "future_stockout_flag_1w",
    "scenario_type",
    "draft_mo_component_status",
]

NUMERIC_FEATURES = [
    "unit_cost",
    "available_qty",
    "reserved_qty",
    "forecasted_qty",
    "rop",
    "inventory_to_rop_ratio",
    "available_minus_rop",
    "n_products_using_item",
    "demand_forecast_qty",
    "open_qty",
    "past_due_qty",
    "days_until_po_delivery",
    "projected_without_po",
    "projected_with_po",
    "po_needed_for_coverage",
    "lead_time_days",
    "vendor_total_completed_pos",
    "vendor_on_time_pos",
    "vendor_late_pos",
    "calculated_vendor_reliability",
    # Raw as-of historical reliability for the vendor on the current open PO.
    # This is prediction-time causal; the deterministic below-75% flag stays excluded.
    "open_po_vendor_reliability",
    "draft_mo_required_qty",
    "draft_mo_shortage_qty",
    "draft_mo_coverage_ratio",
    "consumed_qty",
    "material_overrun_qty",
    "unreserved_available_qty",
    "material_shortfall_qty",
    "consumption_to_reservation_ratio",
]

CATEGORICAL_FEATURES = [
    "item_type",
    "unit_of_measurement",
    "warehouse_id",
    "warehouse_type",
    "draft_mo_status",
    "mo_status",
]

FEATURE_COLUMNS = [
    # Item and warehouse
    "item_type",
    "unit_cost",
    "unit_of_measurement",
    "warehouse_id",
    "warehouse_type",
    # Inventory
    "available_qty",
    "reserved_qty",
    "forecasted_qty",
    "rop",
    "inventory_to_rop_ratio",
    "available_minus_rop",
    "n_products_using_item",
    # Demand and purchasing
    "demand_forecast_qty",
    "open_qty",
    "past_due_qty",
    "days_until_po_delivery",
    "projected_without_po",
    "projected_with_po",
    "po_needed_for_coverage",
    # Vendor
    "lead_time_days",
    "vendor_total_completed_pos",
    "vendor_on_time_pos",
    "vendor_late_pos",
    "calculated_vendor_reliability",
    "open_po_vendor_reliability",
    # Draft manufacturing orders
    "draft_mo_status",
    "draft_mo_required_qty",
    "draft_mo_shortage_qty",
    "draft_mo_coverage_ratio",
    # Production consumption
    "mo_status",
    "consumed_qty",
    "material_overrun_qty",
    "unreserved_available_qty",
    "material_shortfall_qty",
    "consumption_to_reservation_ratio",
]

# Serving contract logged in the MLflow signature. It includes the approved
# feature allowlist plus identifiers, dates and cold-start guardrail inputs;
# the target and every leakage column remain absent.
MODEL_INPUT_COLUMNS = [
    "item_id",
    "snapshot_date",
    "open_po_deadline",
    "months_of_history",
    "use_rule_based",
    "external_risk_pct",
] + FEATURE_COLUMNS

_BOOLEAN_COLUMNS = [
    "stockout_flag",
    "use_rule_based",
    "primary_vendor_below_75pct",
    "open_po_vendor_below_75pct",
    "has_draft_mo_stock_risk",
    "has_material_overrun_risk",
    "has_material_overrun",
    "calculated_has_draft_mo_stock_risk",
    "calculated_has_material_overrun_risk",
    "calculated_open_po_vendor_below_75pct",
    "unreliable_po_coverage",
    "calculated_low_stock_risk",
    "calculated_stockout_risk_alert",
    "business_scenario_injected",
    "business_scenario_positive",
    "business_scenario_holdout",
]

_NON_NEGATIVE_QUANTITY_COLUMNS = [
    "available_qty",
    "reserved_qty",
    "forecasted_qty",
    "rop",
    "demand_forecast_qty",
    "past_due_qty",
    "open_qty",
    "planned_bom_qty",
    "draft_mo_required_qty",
    "draft_mo_shortage_qty",
    "consumed_qty",
    "material_overrun_qty",
    "unreserved_available_qty",
    "material_shortfall_qty",
    "vendor_total_completed_pos",
    "vendor_on_time_pos",
    "vendor_late_pos",
]


@dataclass(frozen=True)
class TemporalDatasetSplit:
    """Chronological train/calibration/test partitions."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    business_scenario_holdout: pd.DataFrame | None = None


class InventoryDatasetBuilder:
    """Load snapshots and create leakage-safe hazard-model features."""

    required_columns = set(OPERATIONAL_REQUIRED_COLUMNS)

    def load(self, path: str | Path | pd.DataFrame = DEFAULT_DATASET_PATH) -> pd.DataFrame:
        frame = path.copy() if isinstance(path, pd.DataFrame) else pd.read_csv(path, low_memory=False)
        missing = sorted(self.required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"Inventory CSV is missing required columns: {missing}")
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
        if frame["snapshot_date"].isna().any():
            raise ValueError("snapshot_date contains invalid or missing values")
        for column, default in ML_AUDIT_COLUMN_DEFAULTS.items():
            if column not in frame:
                frame[column] = default

        for column in [
            *_BOOLEAN_COLUMNS,
            "calculated_stockout_risk_alert",
            "po_needed_for_coverage",
        ]:
            if column in frame:
                frame[column] = self._to_boolean(frame[column])
        if "external_risk_pct" in frame:
            external_risk = pd.to_numeric(frame["external_risk_pct"], errors="coerce")
            if external_risk.isna().any() or not external_risk.between(0.0, 300.0).all():
                raise ValueError("external_risk_pct must contain percentages from 0 to 300")
        if "adjusted_lead_time_days" in frame:
            adjusted = pd.to_numeric(frame["adjusted_lead_time_days"], errors="coerce")
            if (adjusted.dropna() < 0.0).any():
                raise ValueError("adjusted_lead_time_days cannot be negative")

        duplicate_snapshot_id = frame["snapshot_id"].duplicated(keep=False)
        if duplicate_snapshot_id.any():
            ids = frame.loc[duplicate_snapshot_id, "snapshot_id"].head(5).tolist()
            raise ValueError(f"Duplicate snapshot_id values found: {ids}")

        duplicate = frame.duplicated(KEY_COLUMNS + ["snapshot_date"], keep=False)
        if duplicate.any():
            keys = frame.loc[duplicate, KEY_COLUMNS + ["snapshot_date"]].head(5)
            raise ValueError(f"Duplicate item/warehouse/week snapshots found:\n{keys}")

        frame = frame.sort_values(KEY_COLUMNS + ["snapshot_date"]).reset_index(drop=True)
        frame = self.derive_future_outcomes(frame)
        self._validate_business_scenario_data(frame)
        return frame

    @staticmethod
    def derive_future_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
        """Create causal one-week audit labels from observed future snapshots."""
        out = frame.copy()
        grouped = out.groupby(KEY_COLUMNS, sort=False)
        next_date = grouped["snapshot_date"].shift(-1)
        next_event = grouped[EVENT_COLUMN].shift(-1).astype("boolean")
        is_exactly_one_week = next_date.sub(out["snapshot_date"]).dt.days.eq(7)
        out["future_outcome_date_1w"] = next_date.where(is_exactly_one_week)
        out["future_stockout_flag_1w"] = next_event.where(
            is_exactly_one_week
        ).astype("boolean")
        return out

    @staticmethod
    def _validate_business_scenario_data(frame: pd.DataFrame) -> None:
        """Fail closed on invalid enriched business-scenario/audit data."""
        for column in _NON_NEGATIVE_QUANTITY_COLUMNS:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.lt(0.0).any():
                raise ValueError(f"{column} cannot contain negative quantities")

        if "calculated_vendor_reliability" in frame:
            reliability = pd.to_numeric(
                frame["calculated_vendor_reliability"], errors="coerce"
            )
            observed_reliability = reliability.dropna()
            if not observed_reliability.between(0.0, 1.0).all():
                raise ValueError("calculated_vendor_reliability must be between 0 and 1")

        vendor_columns = {
            "vendor_total_completed_pos", "vendor_on_time_pos", "vendor_late_pos"
        }
        if vendor_columns <= set(frame):
            vendor_total = pd.to_numeric(
                frame["vendor_total_completed_pos"], errors="coerce"
            )
            vendor_parts = (
                pd.to_numeric(frame["vendor_on_time_pos"], errors="coerce")
                + pd.to_numeric(frame["vendor_late_pos"], errors="coerce")
            )
            history_present = frame[list(vendor_columns)].notna().any(axis=1)
            history_complete = frame[list(vendor_columns)].notna().all(axis=1)
            if (history_present & ~history_complete).any() or not np.allclose(
                vendor_total[history_complete], vendor_parts[history_complete]
            ):
                raise ValueError(
                    "vendor_on_time_pos + vendor_late_pos must equal vendor_total_completed_pos"
                )

        available = pd.to_numeric(frame["available_qty"], errors="coerce")
        if {"draft_mo_required_qty", "calculated_has_draft_mo_stock_risk"} <= set(frame):
            draft_required = pd.to_numeric(
                frame["draft_mo_required_qty"], errors="coerce"
            )
            draft_risk = frame["calculated_has_draft_mo_stock_risk"]
            if not draft_required[draft_risk].gt(available[draft_risk]).all():
                raise ValueError("Draft-risk rows must require more than available quantity")

        if {"material_shortfall_qty", "calculated_has_material_overrun_risk"} <= set(frame):
            material_shortfall = pd.to_numeric(
                frame["material_shortfall_qty"], errors="coerce"
            )
            material_risk = frame["calculated_has_material_overrun_risk"]
            if not material_shortfall[material_risk].gt(0.0).all():
                raise ValueError("Material-overrun-risk rows must have a positive shortfall")

        open_po_reliability = pd.to_numeric(
            frame["open_po_vendor_reliability"], errors="coerce"
        )
        if "unreliable_po_coverage" in frame:
            unreliable_coverage = frame["unreliable_po_coverage"]
            if not open_po_reliability[unreliable_coverage].lt(0.75).all():
                raise ValueError(
                    "Unreliable PO coverage requires open-PO reliability below 0.75"
                )

        injected = frame["business_scenario_injected"]
        injected_dates = frame.loc[injected, "future_outcome_date_1w"]
        injected_outcomes = frame.loc[injected, "future_stockout_flag_1w"]
        if injected_dates.isna().any() or injected_outcomes.isna().any():
            raise ValueError("Injected scenarios require a valid one-week future outcome")
        expected_dates = frame.loc[injected, "snapshot_date"] + pd.Timedelta(days=7)
        if not injected_dates.eq(expected_dates).all():
            raise ValueError("Injected future_outcome_date_1w must be exactly one week later")

        future_lookup = frame.set_index(KEY_COLUMNS + ["snapshot_date"])[EVENT_COLUMN]
        future_keys = pd.MultiIndex.from_frame(
            frame.loc[injected, KEY_COLUMNS].assign(snapshot_date=injected_dates.to_numpy())
        )
        actual_future = future_lookup.reindex(future_keys)
        if actual_future.isna().any():
            raise ValueError("Injected scenarios must have a matching future snapshot")
        expected_future = injected_outcomes.astype(bool).to_numpy()
        if not np.array_equal(actual_future.astype(bool).to_numpy(), expected_future):
            raise ValueError("Injected future outcome does not match the future snapshot")
        invalid_origins = injected & (
            frame["stockout_flag"].astype(bool)
            | pd.to_numeric(frame["available_qty"], errors="coerce").le(0.0)
        )
        if invalid_origins.any():
            raise ValueError("Injected scenario origins must be positive-stock, non-stockout rows")

    def build_supervised_frame(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Build a conditional one-week hazard target and horizon outcomes.

        Rows already stocked out at the origin are outside the risk set. The
        final observation for an item/warehouse is censored because its next
        week's outcome is unobserved, rather than being treated as a negative.
        """
        frame = snapshots.copy().sort_values(KEY_COLUMNS + ["snapshot_date"])
        grouped = frame.groupby(KEY_COLUMNS, sort=False)[EVENT_COLUMN]
        grouped_dates = frame.groupby(KEY_COLUMNS, sort=False)["snapshot_date"]
        future = [grouped.shift(-week).astype("boolean") for week in range(1, 10)]
        future_dates = [grouped_dates.shift(-week) for week in range(1, 10)]
        observed_at_interval = [
            (future_date - frame["snapshot_date"]).dt.days.eq(7 * week)
            for week, future_date in enumerate(future_dates, start=1)
        ]
        frame["target"] = future[0].where(observed_at_interval[0])

        for horizon, weeks in ((4, future[:4]), (9, future)):
            valid = pd.concat(weeks, axis=1).notna().all(axis=1)
            valid &= pd.concat(observed_at_interval[:horizon], axis=1).all(axis=1)
            outcome = pd.concat(weeks, axis=1).fillna(False).any(axis=1)
            frame[f"target_{horizon}w"] = outcome.where(valid).astype("boolean")

        valid_target = frame["target"].notna() & ~frame[EVENT_COLUMN].astype(bool)
        frame = frame[valid_target].copy()
        frame["target"] = frame["target"].astype(int)
        engineered = self.engineer_features(frame)
        leaked = sorted(set(FEATURE_COLUMNS) & set(EXCLUDED_LEAKAGE_COLUMNS))
        if leaked:  # defensive invariant: fail closed if the allowlist changes.
            raise RuntimeError(f"Leakage columns entered the model feature list: {leaked}")
        return engineered

    def engineer_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Create the approved direct and Prisma-derived model features."""
        out = frame.copy()
        for column in NUMERIC_FEATURES:
            if column not in out:
                out[column] = np.nan
        for column in CATEGORICAL_FEATURES:
            if column not in out:
                out[column] = "unknown"

        for column in NUMERIC_FEATURES:
            out[column] = pd.to_numeric(out[column], errors="coerce")

        # Open-PO metadata is meaningful only while quantity is still outstanding.
        # Apply this invariant in shared feature engineering so historical training
        # rows and projected serving rows use identical missing-value semantics.
        active_open_po = out["open_qty"].fillna(0.0).gt(0.0)
        deadline_source = (
            out["open_po_deadline"]
            if "open_po_deadline" in out
            else pd.Series(pd.NaT, index=out.index)
        )
        open_po_deadline = pd.to_datetime(deadline_source, errors="coerce").where(
            active_open_po
        )
        out["open_po_deadline"] = open_po_deadline
        out["open_po_vendor_reliability"] = out[
            "open_po_vendor_reliability"
        ].where(active_open_po)

        external_risk = self._numeric_or_default(out, "external_risk_pct")
        # ``adjusted_lead_time_days`` is deliberately derived here rather than
        # accepted from prediction clients. This keeps training and serving on
        # one formula and prevents stale/tampered derived values entering the model.
        if _BASE_LEAD_TIME_COLUMN not in out:
            out[_BASE_LEAD_TIME_COLUMN] = out["lead_time_days"].copy()
        base_lead_time = pd.to_numeric(
            out[_BASE_LEAD_TIME_COLUMN], errors="coerce"
        )
        adjusted_lead_time = base_lead_time * (1.0 + external_risk / 100.0)
        out["adjusted_lead_time_days"] = adjusted_lead_time
        out["lead_time_days"] = adjusted_lead_time.where(
            external_risk.gt(0.0), base_lead_time
        )

        # Prefer the explicitly calculated vendor history; older exports can
        # still fall back to their historical ``computed_reliability`` field.
        if "computed_reliability" in out:
            out["calculated_vendor_reliability"] = out[
                "calculated_vendor_reliability"
            ].fillna(pd.to_numeric(out["computed_reliability"], errors="coerce"))

        weekly_demand = (
            self._numeric_or_default(out, "demand_forecast_qty")
            / DEMAND_FORECAST_WEEKS
        )
        out["weekly_demand_qty"] = weekly_demand
        projected_week_demand = weekly_demand.copy()
        projected_week_demand += self._numeric_or_default(out, "planned_bom_qty")
        projected_week_demand += self._numeric_or_default(out, "past_due_qty")

        rop = out["rop"]
        out["inventory_to_rop_ratio"] = self._safe_ratio(out["available_qty"], rop)
        out["available_minus_rop"] = out["available_qty"] - rop
        out["projected_without_po"] = out["available_qty"] - projected_week_demand
        out["projected_with_po"] = (
            out["available_qty"] + out["open_qty"] - projected_week_demand
        )
        out["po_needed_for_coverage"] = (
            active_open_po & (out["projected_without_po"] < rop)
        ).astype(float)
        out["draft_mo_shortage_qty"] = (
            out["draft_mo_required_qty"] - out["available_qty"]
        ).clip(lower=0.0)
        out["draft_mo_coverage_ratio"] = self._safe_ratio(
            out["available_qty"], out["draft_mo_required_qty"]
        )
        out["material_overrun_qty"] = (
            out["consumed_qty"] - out["reserved_qty"]
        ).clip(lower=0.0)
        out["unreserved_available_qty"] = (
            out["available_qty"] - out["reserved_qty"]
        ).clip(lower=0.0)
        out["material_shortfall_qty"] = (
            out["material_overrun_qty"] - out["unreserved_available_qty"]
        ).clip(lower=0.0)
        out["consumption_to_reservation_ratio"] = self._safe_ratio(
            out["consumed_qty"], out["reserved_qty"]
        )
        snapshot_date = pd.to_datetime(out["snapshot_date"], errors="coerce")
        out["days_until_po_delivery"] = (
            open_po_deadline - snapshot_date
        ).dt.days.astype(float)

        return out

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Divide without emitting infinities for zero or missing denominators."""
        den = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
        num = pd.to_numeric(numerator, errors="coerce")
        return num.div(den).replace([np.inf, -np.inf], np.nan)

    def project_week(self, frame: pd.DataFrame, week_index: int) -> pd.DataFrame:
        """Project an origin snapshot to the start of a future hazard interval.

        ``week_index=0`` is the next-week hazard using the current snapshot.
        Later intervals consume the weekly demand forecast. An open PO is added
        once its deadline has passed, then removed from remaining incoming cover.
        """
        if week_index < 0:
            raise ValueError("week_index must be non-negative")
        projected = frame.copy()
        origin_date = pd.to_datetime(projected["snapshot_date"], errors="coerce")
        projected["snapshot_date"] = origin_date + pd.to_timedelta(7 * week_index, unit="D")

        demand = (
            self._numeric_or_default(projected, "demand_forecast_qty")
            / DEMAND_FORECAST_WEEKS
        )
        demand += self._numeric_or_default(projected, "planned_bom_qty")
        demand += self._numeric_or_default(projected, "past_due_qty")
        available = pd.to_numeric(projected["available_qty"], errors="coerce").fillna(0.0)
        incoming = self._numeric_or_default(projected, "open_qty")
        deadline_source = (
            projected["open_po_deadline"]
            if "open_po_deadline" in projected
            else pd.Series(pd.NaT, index=projected.index)
        )
        deadline = pd.to_datetime(deadline_source, errors="coerce")
        receipt_has_arrived = deadline.notna() & (deadline <= projected["snapshot_date"])
        projected["available_qty"] = (
            available - demand * week_index + incoming.where(receipt_has_arrived, 0.0)
        ).clip(lower=0.0)
        projected["open_qty"] = incoming.where(~receipt_has_arrived, 0.0)
        projected["open_po_deadline"] = deadline.where(~receipt_has_arrived)
        if "open_po_vendor_reliability" in projected:
            reliability = pd.to_numeric(
                projected["open_po_vendor_reliability"], errors="coerce"
            )
            projected["open_po_vendor_reliability"] = reliability.where(
                ~receipt_has_arrived
            )
        return self.engineer_features(projected)

    def project_week_projection_based(
        self, frame: pd.DataFrame, week_index: int
    ) -> pd.DataFrame:
        """Project a richer expected future state for comparison scoring.

        This path is deliberately separate from ``project_week`` so the current
        production/horizon-calibrated behavior remains unchanged. Forecast
        demand recurs weekly; known Draft-MO and active-MO quantities are applied
        once after the origin; reservations persist; and an expected PO receipt
        is reliability-weighted at its scheduled arrival date.
        """
        if week_index < 0:
            raise ValueError("week_index must be non-negative")
        projected = self.engineer_features(frame).copy()
        origin_date = pd.to_datetime(projected["snapshot_date"], errors="coerce")
        projected_date = origin_date + pd.to_timedelta(7 * week_index, unit="D")
        projected["snapshot_date"] = projected_date

        weekly_demand = (
            self._numeric_or_default(projected, "demand_forecast_qty")
            / DEMAND_FORECAST_WEEKS
        )
        recurring_demand = weekly_demand * week_index
        apply_known_once = float(week_index > 0)
        past_due = self._numeric_or_default(projected, "past_due_qty")
        planned_bom = self._numeric_or_default(projected, "planned_bom_qty")

        draft_status = projected["draft_mo_status"].astype("string").str.strip().str.casefold()
        draft_demand = self._numeric_or_default(
            projected, "draft_mo_required_qty"
        ).where(draft_status.eq("draft"), 0.0)

        mo_status = projected["mo_status"].astype("string").str.strip().str.casefold()
        active_mo = mo_status.isin({"confirmed", "in progress"})
        active_consumption = self._numeric_or_default(
            projected, "consumed_qty"
        ).where(active_mo, 0.0)
        known_one_time_demand = (
            past_due + planned_bom + draft_demand + active_consumption
        ) * apply_known_once

        available = self._numeric_or_default(projected, "available_qty")
        incoming = self._numeric_or_default(projected, "open_qty")
        deadline = pd.to_datetime(projected["open_po_deadline"], errors="coerce")
        receipt_has_arrived = deadline.notna() & deadline.le(projected_date)
        reliability = pd.to_numeric(
            projected["open_po_vendor_reliability"], errors="coerce"
        ).clip(lower=0.0, upper=1.0)
        expected_receipt = incoming * reliability.fillna(1.0)

        projected["available_qty"] = (
            available
            - recurring_demand
            - known_one_time_demand
            + expected_receipt.where(receipt_has_arrived, 0.0)
        ).clip(lower=0.0)
        # Reserved stock remains committed throughout the projection. It is not
        # repeatedly deducted from physical available quantity.
        projected["reserved_qty"] = self._numeric_or_default(
            projected, "reserved_qty"
        )
        projected["open_qty"] = incoming.where(~receipt_has_arrived, 0.0)
        projected["open_po_deadline"] = deadline.where(~receipt_has_arrived)
        projected["open_po_vendor_reliability"] = reliability.where(
            ~receipt_has_arrived
        )
        return self.engineer_features(projected)

    @staticmethod
    def _numeric_or_default(
        frame: pd.DataFrame, column: str, default: float = 0.0
    ) -> pd.Series:
        values = frame[column] if column in frame else pd.Series(default, index=frame.index)
        return pd.to_numeric(values, errors="coerce").fillna(default)

    def split_for_training(
        self,
        frame: pd.DataFrame,
        train_fraction: float = 0.70,
        calibration_fraction: float = 0.15,
        random_state: int = 42,
    ) -> TemporalDatasetSplit:
        """Create fully chronological train, calibration and test windows.

        Whole snapshot dates—not individual rows—are assigned to partitions.
        Calibration therefore follows training in time and the newest 15% of
        weeks remain an untouched out-of-time test. ``random_state`` is retained
        for API compatibility but this split is intentionally deterministic.
        """
        if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
            raise ValueError("split fractions must be between zero and one")
        if train_fraction + calibration_fraction >= 1:
            raise ValueError("train_fraction + calibration_fraction must be below one")

        dates = np.array(sorted(pd.to_datetime(frame["snapshot_date"]).unique()))
        if len(dates) < 10:
            raise ValueError("At least 10 distinct weekly snapshots are required")
        train_index = max(1, int(len(dates) * train_fraction))
        calibration_index = max(
            train_index + 1,
            int(len(dates) * (train_fraction + calibration_fraction)),
        )
        calibration_index = min(calibration_index, len(dates) - 1)
        calibration_start = dates[train_index]
        test_start = dates[calibration_index]
        holdout_marker = frame.get(
            "business_scenario_holdout", pd.Series(False, index=frame.index)
        ).astype("boolean").fillna(False).astype(bool)
        if (holdout_marker & frame["snapshot_date"].ge(test_start)).any():
            raise ValueError("Business-scenario holdout rows must remain before the future test")
        holdout_groups = pd.MultiIndex.from_frame(
            frame.loc[holdout_marker, KEY_COLUMNS].astype(str).drop_duplicates()
        )
        row_groups = pd.MultiIndex.from_frame(frame[KEY_COLUMNS].astype(str))
        grouped_holdout = pd.Series(row_groups.isin(holdout_groups), index=frame.index)
        historical_holdout = grouped_holdout & frame["snapshot_date"].lt(test_start)

        train = frame[
            (frame["snapshot_date"] < calibration_start) & ~historical_holdout
        ].copy()
        calibration = frame[
            (frame["snapshot_date"] >= calibration_start)
            & (frame["snapshot_date"] < test_start)
            & ~historical_holdout
        ].copy()
        test = frame[frame["snapshot_date"] >= test_start].copy()
        business_scenario_holdout = frame[holdout_marker].copy()
        for name, part in (("train", train), ("calibration", calibration), ("test", test)):
            if part.empty or part["target"].nunique() < 2:
                raise ValueError(f"{name} split must contain both target classes")
        return TemporalDatasetSplit(
            train=train,
            calibration=calibration,
            test=test,
            business_scenario_holdout=business_scenario_holdout,
        )

    def split_by_time(
        self,
        frame: pd.DataFrame,
        train_fraction: float = 0.70,
        calibration_fraction: float = 0.15,
        random_state: int = 42,
    ) -> TemporalDatasetSplit:
        """Backward-compatible alias for :meth:`split_for_training`."""
        return self.split_for_training(
            frame,
            train_fraction=train_fraction,
            calibration_fraction=calibration_fraction,
            random_state=random_state,
        )

    @staticmethod
    def _to_boolean(values: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(values):
            return values.astype(bool)
        normalized = values.astype("string").str.strip().str.lower()
        mapped = normalized.map(
            {
                "true": True,
                "1": True,
                "1.0": True,
                "yes": True,
                "false": False,
                "0": False,
                "0.0": False,
                "no": False,
            }
        )
        if mapped.isna().any():
            bad = sorted(normalized[mapped.isna()].dropna().unique().tolist())
            raise ValueError(f"Invalid boolean values: {bad[:5]}")
        return mapped.astype(bool)

    @staticmethod
    def _to_nullable_boolean(values: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(values.dtype):
            return values.astype("boolean")
        normalized = values.astype("string").str.strip().str.lower()
        mapped = normalized.map(
            {
                "true": True,
                "1": True,
                "1.0": True,
                "yes": True,
                "false": False,
                "0": False,
                "0.0": False,
                "no": False,
            }
        )
        invalid = normalized.notna() & mapped.isna()
        if invalid.any():
            bad = sorted(normalized[invalid].unique().tolist())
            raise ValueError(f"Invalid nullable boolean values: {bad[:5]}")
        return mapped.astype("boolean")
