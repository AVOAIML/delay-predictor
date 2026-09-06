"""Deterministic inventory alert rules kept separate from ML probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LOW_STOCK = "low_stock"
DRAFT_MO_RISK = "draft_mo_risk"
MATERIAL_OVERRUN_RISK = "material_overrun_risk"
UNRELIABLE_PO_COVERAGE = "unreliable_po_coverage"
BUSINESS_RULE_NAMES = (
    LOW_STOCK,
    DRAFT_MO_RISK,
    MATERIAL_OVERRUN_RISK,
    UNRELIABLE_PO_COVERAGE,
)

# --- Alerting thresholds ----------------------------------------------------
# Every number that drives an alert, suppression, or fallback decision across
# M2 lives here, in one place, so it's easy to find, document, and change
# consistently instead of drifting across files (see hazard_model_base.py and
# db_prediction.py, which import these rather than hardcoding their own copy).

# Below this on-time-delivery fraction, an open PO is not trustworthy cover for
# a projected shortfall (per product's vendor-reliability user story). Shared
# with inventory_dataset.py's business-scenario validator so both stay in sync.
UNRELIABLE_PO_RELIABILITY_THRESHOLD = 0.75

# Above this on-time-delivery fraction, an open PO covering the projected
# deficit is trusted enough to suppress the model's alert
# (BaseInventoryHazardModel.predict_risk's ``suppressed`` column).
SUPPRESSION_RELIABILITY_THRESHOLD = 0.95

# Above this calibrated 30-day stockout probability, the model itself raises
# an alert, independent of the deterministic rules above. Also the boundary
# between the "Green" and "Amber" risk badges shown in the UI
# (BaseInventoryHazardModel._badge) — the two intentionally share this
# constant so tuning one can't silently desync the other.
MODEL_ALERT_THRESHOLD = 0.33

# At or above this calibrated risk, the badge shown is "Red" instead of
# "Amber". Below MODEL_ALERT_THRESHOLD it's "Green".
BADGE_RED_THRESHOLD = 0.66

# Items with fewer than this many months of observed history fall back to the
# deterministic rule-based hazard estimate instead of the trained model —
# too little history for the model's features to be trustworthy. Missing
# history is treated as unknown, not automatically below this line
# (BaseInventoryHazardModel._rule_based_mask).
RULE_BASED_HISTORY_MONTHS = 6.0


@dataclass(frozen=True)
class BusinessRuleEvaluation:
    """Serializable result for one inventory snapshot."""

    override: bool
    flags: dict[str, bool]
    reasons: list[str]


class InventoryBusinessRuleEvaluator:
    """Evaluate deterministic alert policy without changing model probabilities."""

    @classmethod
    def evaluate(cls, frame: pd.DataFrame) -> list[BusinessRuleEvaluation]:
        available = cls._numeric(frame, "available_qty")
        rop = cls._numeric(frame, "rop")
        draft_required = cls._numeric(frame, "draft_mo_required_qty")
        material_shortfall = cls._numeric(frame, "material_shortfall_qty")
        open_quantity = cls._numeric(frame, "open_qty")
        projected_without_po = cls._numeric(frame, "projected_without_po")
        open_po_reliability = cls._numeric(frame, "open_po_vendor_reliability")
        draft_status = cls._status(frame, "draft_mo_status")
        mo_status = cls._status(frame, "mo_status")

        masks = {
            LOW_STOCK: available.lt(rop),
            DRAFT_MO_RISK: draft_status.eq("draft") & draft_required.gt(available),
            MATERIAL_OVERRUN_RISK: (
                mo_status.isin({"confirmed", "in progress"})
                & material_shortfall.gt(0.0)
            ),
            UNRELIABLE_PO_COVERAGE: (
                open_quantity.gt(0.0)
                & projected_without_po.lt(rop)
                # Unknown reliability is a bigger unknown than a proven-flaky
                # vendor, not a smaller one — treat it as unreliable rather
                # than silently assuming the vendor is fine.
                & (
                    open_po_reliability.lt(UNRELIABLE_PO_RELIABILITY_THRESHOLD)
                    | open_po_reliability.isna()
                )
            ),
        }

        evaluations = []
        for position in range(len(frame)):
            flags = {
                name: bool(mask.iloc[position])
                for name, mask in masks.items()
            }
            reasons = [name for name in BUSINESS_RULE_NAMES if flags[name]]
            evaluations.append(BusinessRuleEvaluation(bool(reasons), flags, reasons))
        return evaluations

    @staticmethod
    def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        values = frame[column] if column in frame else pd.Series(float("nan"), index=frame.index)
        return pd.to_numeric(values, errors="coerce")

    @staticmethod
    def _status(frame: pd.DataFrame, column: str) -> pd.Series:
        values = frame[column] if column in frame else pd.Series("", index=frame.index)
        return values.astype("string").fillna("").str.strip().str.casefold()
