"""Parity-critical DAL transforms (plan §1a, §12a). Pure functions, fully
unit-tested without a database. The DAL applies these between the raw SQL read
and the bronze layer so no module can re-implement them inconsistently.
"""

from __future__ import annotations

import datetime as _dt
from typing import Iterable, Sequence

import pandas as pd

from maxxflow_core.hashing import pseudonymize, skill_tier_from_ratio
from maxxflow_core.money import D

# Columns that DO NOT EXIST at scoring time T for an in-flight MO (M3 leakage).
# The assert_no_leakage test (tests/unit/test_m3_leakage.py) checks the M3 feature
# frame contains none of these (plan §4 M3, §12a leakage discipline).
M3_FORBIDDEN_AT_T = (
    "real_duration", "actual_start", "actual_end", "units_done_final",
    "terminal_status_id", "completed_at",
)


# --- ROP: derive live; NEVER read items.rop_status (plan §1a row 3) ---------
def derive_rop_state(available, rop) -> str:
    a = D(available)
    r = D(rop if rop is not None else 0)
    if a > r:
        return "Available"
    if a < r:
        return "Below ROP"
    return "ROP Reached"


def rop_deficit(available, rop) -> float:
    """Units below ROP (>=0). The M2 risk signal; 0 when at/above ROP."""
    a = D(available)
    r = D(rop if rop is not None else 0)
    deficit = r - a
    return float(deficit) if deficit > 0 else 0.0


# --- soft deletes (plan §1a last row) ---------------------------------------
def filter_active(df: pd.DataFrame, col: str = "deleted_at") -> pd.DataFrame:
    if col not in df.columns:
        return df
    return df[df[col].isna()].copy()


# --- operator PII -> pseudonymous token BEFORE bronze (plan §1a, §12a #9) ----
def pseudonymize_operators(df: pd.DataFrame, columns: Sequence[str], salt: str) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        def _map(v):
            if isinstance(v, (list, tuple)):
                return [pseudonymize(x, salt) for x in v]
            return pseudonymize(v, salt)
        out[col] = out[col].map(_map)
    return out


def operator_skill_tier(real_expected_ratio) -> int:
    return skill_tier_from_ratio(real_expected_ratio)


# --- WorkCenter capacity = len(allowedEmployees) w/ 0-guard (§1a, §12a #5) ---
def work_center_capacity(allowed_employees) -> int | None:
    if allowed_employees is None:
        return None
    try:
        n = len(allowed_employees)
    except TypeError:
        return None
    return n if n > 0 else None  # empty => "unknown", drop the ratio (never /0)


def concurrency_ratio(active_count: int, capacity: int | None) -> float | None:
    if not capacity:  # None or 0
        return None
    return active_count / capacity


# --- GRN vendor on-time keyed off STATUS TRANSITION (plan §1a, §12a #4) ------
_RECEIVED_STATUSES = {"Goods Received", "Partially Received"}


def grn_on_time(status: str | None, scheduled_delivery_date, transition_at) -> bool | None:
    """on-time iff the GRN transitioned to received AT/BEFORE the scheduled date.

    Keys off the status transition timestamp (``updated_at`` / ``bill_created_at``),
    NOT ``created_at``. Null ``scheduled_delivery_date`` => None ("unknown"):
    excluded from the reliability denominator, never counted as late.
    """
    if scheduled_delivery_date is None or pd.isna(scheduled_delivery_date):
        return None
    if status not in _RECEIVED_STATUSES or transition_at is None or pd.isna(transition_at):
        return None
    return transition_at <= scheduled_delivery_date


def vendor_reliability(on_time_flags: Iterable[bool | None]) -> float | None:
    """Fraction on-time, EXCLUDING unknowns (None) from the denominator."""
    known = [b for b in on_time_flags if b is not None]
    if not known:
        return None
    return sum(1 for b in known if b) / len(known)


# --- M3 as-of-T leakage censoring (plan §4 M3, §12a) ------------------------
def censor_timelogs_as_of(time_logs: pd.DataFrame, t: _dt.datetime,
                          started_col: str = "started_at",
                          ended_col: str = "ended_at") -> float:
    """Minutes worked as-of T: keep logs with started_at <= T, clip ended_at to T
    (running logs end at T), recompute the clipped duration. No future leakage."""
    if time_logs.empty:
        return 0.0
    df = time_logs[time_logs[started_col] <= t].copy()
    if df.empty:
        return 0.0
    ended = df[ended_col].where(df[ended_col].notna(), t)
    ended = ended.where(ended <= t, t)  # clip to T
    minutes = (pd.to_datetime(ended) - pd.to_datetime(df[started_col])).dt.total_seconds() / 60.0
    return float(minutes.clip(lower=0).sum())


def physical_fraction(units_done, quantity) -> float:
    q = D(quantity)
    if q <= 0:
        return 0.0
    return float(D(units_done) / q)
