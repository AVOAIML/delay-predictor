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
                & open_po_reliability.lt(0.75)
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
