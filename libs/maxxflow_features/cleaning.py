"""Generic, module-agnostic data cleaning for every training path (M1..M4).

Only the GENERIC mechanics live here — type coercion, duplicate-row removal,
outlier winsorising and missing-value imputation. **Domain semantics stay in each
module's feature-contract code** (for example M2's ``inventory_dataset.py`` and
M4's feature builder). A single generic imputer would get those domain defaults
wrong, so this module deliberately does not try.

Design notes
------------
* **fit / apply split.** ``fit_*`` computes statistics, ``apply_*`` uses them. This
  lets a caller fit on the TRAIN split and apply the same numbers to holdout and to
  serving, which is how train/serve skew is avoided. ``clean_frame`` accepts a
  pre-fitted ``stats`` bundle for exactly that reason.
* **Winsorise, never drop.** Outliers are clipped to a wide quantile band, not
  removed: dropping rows loses labels, and clipping is enough to stop one absurd
  value from shifting the imputation median or dominating a linear combiner.
  Tree models split on order, so clipping preserves "this row is extreme".
* **Duplicate rows are dropped, not flagged.** Exact duplicates carry no extra
  information and silently double-weight a pattern. (Distinct from M4's *business*
  duplicate rule in ``m4_bom/rules.py``, which detects duplicate BOM lines as a
  predictive FEATURE — that is a different thing and stays where it is.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# Conservative band: touches ~1% of rows, so genuine signal is preserved.
LOWER_Q = 0.005
UPPER_Q = 0.995


@dataclass
class CleaningReport:
    """Counts for the training log, the UI and MLflow tags (data-quality lineage)."""
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    missing_filled: int = 0
    outliers_clipped: int = 0
    columns_cleaned: int = 0
    stats: dict = field(default_factory=dict)   # fitted medians/modes/bounds

    def as_tags(self) -> dict[str, str]:
        """Flatten to MLflow tag strings so each model version records its data quality."""
        return {
            "dq_rows_in": str(self.rows_in),
            "dq_rows_out": str(self.rows_out),
            "dq_duplicates_removed": str(self.duplicates_removed),
            "dq_missing_filled": str(self.missing_filled),
            "dq_outliers_clipped": str(self.outliers_clipped),
        }

    def summary(self) -> str:
        return (f"rows {self.rows_in}->{self.rows_out}, "
                f"{self.duplicates_removed} duplicate rows removed, "
                f"{self.missing_filled} missing values filled, "
                f"{self.outliers_clipped} outlier values clipped")


# --- duplicates -------------------------------------------------------------
def drop_duplicate_rows(df: pd.DataFrame, subset: Sequence[str] | None = None,
                        logger: Any = None) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows (keeping the first).

    ``subset`` should normally be the feature columns PLUS the label: two rows with
    identical features and the same outcome are redundant, whereas identical features
    with *different* outcomes are real label noise and must be kept.
    """
    if df.empty:
        return df, 0
    cols = [c for c in (subset or df.columns) if c in df.columns]
    if not cols:
        return df, 0
    mask = df.duplicated(subset=cols, keep="first")
    n = int(mask.sum())
    if n and logger:
        logger.log(f"Cleaning: dropped {n} exact duplicate row(s) on {len(cols)} column(s)")
    return (df.loc[~mask].copy() if n else df), n


# --- outliers ---------------------------------------------------------------
def fit_outlier_bounds(df: pd.DataFrame, numeric: Sequence[str],
                       lower_q: float = LOWER_Q, upper_q: float = UPPER_Q) -> dict[str, tuple[float, float]]:
    """Quantile bounds per numeric column. Constant/empty columns are skipped."""
    bounds: dict[str, tuple[float, float]] = {}
    for c in numeric:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        lo, hi = float(s.quantile(lower_q)), float(s.quantile(upper_q))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            bounds[c] = (lo, hi)
    return bounds


def apply_outlier_bounds(df: pd.DataFrame, bounds: Mapping[str, tuple[float, float]],
                         logger: Any = None) -> tuple[pd.DataFrame, int]:
    """Clip each column into its fitted band. Returns the frame + values clipped."""
    if not bounds:
        return df, 0
    out = df.copy()
    clipped = 0
    for c, (lo, hi) in bounds.items():
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        clipped += int(((s < lo) | (s > hi)).sum())
        out[c] = s.clip(lo, hi)
    if clipped and logger:
        logger.log(f"Cleaning: clipped {clipped} outlier value(s) to the "
                   f"{LOWER_Q:.1%}-{UPPER_Q:.1%} band across {len(bounds)} column(s)")
    return out, clipped


# --- missing values ---------------------------------------------------------
def fit_impute_stats(df: pd.DataFrame, numeric: Sequence[str],
                     categorical: Sequence[str]) -> dict[str, Any]:
    """Median per numeric column; the literal 'unknown' sentinel for categoricals.

    Persist this with the model so serving fills gaps the SAME way training did.
    """
    med: dict[str, float] = {}
    for c in numeric:
        if c in df.columns:
            m = pd.to_numeric(df[c], errors="coerce").median()
            med[c] = float(m) if pd.notna(m) else 0.0
    return {"medians": med, "categorical_fill": {c: "unknown" for c in categorical if c in df.columns}}


def apply_impute_stats(df: pd.DataFrame, stats: Mapping[str, Any],
                       logger: Any = None) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    filled = 0
    for c, m in (stats.get("medians") or {}).items():
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        filled += int(s.isna().sum())
        out[c] = s.fillna(m)
    for c, fill in (stats.get("categorical_fill") or {}).items():
        if c not in out.columns:
            continue
        filled += int(out[c].isna().sum())
        out[c] = out[c].astype("object").where(out[c].notna(), fill).astype("category")
    if filled and logger:
        logger.log(f"Cleaning: filled {filled} missing value(s)")
    return out, filled


# --- one-stop generic pass --------------------------------------------------
def clean_frame(df: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str],
                logger: Any = None, *, dedupe_subset: Sequence[str] | None = None,
                winsorize: bool | Sequence[str] = True,
                stats: Mapping[str, Any] | None = None,
                ) -> tuple[pd.DataFrame, CleaningReport]:
    """Generic pass: DEDUPE -> WINSORISE -> IMPUTE, in that order.

    Order matters: duplicates first (so they don't skew the quantiles), outliers next
    (so an absurd value can't drag the imputation median), imputation last.

    ``winsorize`` is ``True`` (clip every numeric column), ``False`` (clip none), or an
    explicit LIST of columns to clip. The list form exists because a column can be one
    whose tails ARE the signal — M1's ``price_ratio`` is the standing example: the deep
    discounts and premium prices at the extremes are exactly the decisions the model is
    asked about, and that column carries its own trained-range guardrail already. Pass
    the list rather than clipping it and losing the ends.

    Pass ``stats`` (from a previous ``CleaningReport.stats``) to reuse fitted
    bounds/medians instead of re-deriving them — use that for the holdout and at
    serving time. Returns the cleaned frame and a :class:`CleaningReport`.
    """
    rep = CleaningReport(rows_in=int(len(df)))
    present_num = [c for c in numeric if c in df.columns]
    present_cat = [c for c in categorical if c in df.columns]
    out = df

    out, rep.duplicates_removed = drop_duplicate_rows(
        out, dedupe_subset or (list(present_num) + list(present_cat)), logger)

    fitted: dict[str, Any] = dict(stats or {})
    if winsorize is not False:
        clip_cols = (present_num if winsorize is True
                     else [c for c in winsorize if c in out.columns])
        bounds = fitted.get("outlier_bounds")
        if bounds is None:
            bounds = fit_outlier_bounds(out, clip_cols)
            fitted["outlier_bounds"] = bounds
        out, rep.outliers_clipped = apply_outlier_bounds(out, bounds, logger)

    imp = fitted.get("impute")
    if imp is None:
        imp = fit_impute_stats(out, present_num, present_cat)
        fitted["impute"] = imp
    out, rep.missing_filled = apply_impute_stats(out, imp, logger)

    rep.rows_out = int(len(out))
    rep.columns_cleaned = len(present_num) + len(present_cat)
    rep.stats = fitted
    return out, rep
