"""Delay-risk elements computed FROM a job dict produced by
`rollup.build_job_rollups()` — no DB access, no raw tables, pure transform
over that object (`predecessor_time_overrun_ratio` relies on
`depends_on_operation_ids`, which `rollup.py` populates from
`operation_dependencies` for exactly this reason).

`material_shortfall_ratio` deliberately matches the driver name
`maxxflow_synth/ground_truth.py`'s `M3Weights`/`m3_delay_logit` already use,
so this module's output can feed that model's inputs directly, with the same
vocabulary, later.

Each element, and exactly how it reads the rollup shape:

  time_overrun_ratio (per operation)
    = actual_duration_minutes / (work_done_percentage * expected_duration_minutes)
    where work_done_percentage = current_done_quantity / job_quantity.
    Weighs the logged time against how much of the job is ACTUALLY done, not
    the full expected duration - an operation 60% done in 350 minutes against
    a 480-minute budget has technically used LESS time than the full budget
    so far, but at that per-unit rate it's already running behind
    (350 / (0.6*480) ≈ 1.22), which plain actual/expected (350/480 ≈ 0.73)
    would miss until the operation is already overdue. >1 means running
    behind at the current pace; <1 means ahead. None when
    actual_duration_minutes is missing/0 (that field is only set once a work
    order's first time log closes, per dal.py), expected_duration_minutes is
    0, or work_done_percentage is missing/0 (no progress yet to weigh the
    logged time against - e.g. still in a setup/prep phase before the first
    unit is finished).

  predecessor_time_overrun_ratio (per operation)
    = MAX(time_overrun_ratio of each critical-path predecessor)
    Only meaningful for a DEPENDENT operation - i.e. one with at least one
    entry in `depends_on_operation_ids`. None for an operation with no
    predecessors (independent), and None if every predecessor's own
    time_overrun_ratio is itself None (nothing to take a max of).

  operator_pace_ratio (per operation)
    = AVG over the operation's assigned operators of R(op)
    R(op) = avg, over that operator's last_10_work_orders that are actually
    COMPLETE (completed_on is not None - an in-progress WO has no elapsed
    time worth comparing to its schedule yet), of
    (elapsed_time_minutes / scheduled_time_minutes).
    An operator with zero completed history contributes nothing to the
    operator_pace_ratio average (excluded, not treated as a 0 or R(op)=1).

  material_shortfall_ratio (per operation)
    = SUM, over the operation's components where
      available_quantity < required_quantity, of
      (required_quantity / available_quantity).
    0.0 when no component is short (empty sum). A short component with
    available_quantity == 0 contributes math.inf (there is literally no
    stock to compute a finite ratio against) rather than raising ZeroDivisionError.

  vendor_lead_time_ratio (per component's vendor)
    = AVG, over that vendor's last_10_purchase_orders, of
      (grn_received_date - po_order_date) / (po_order_deadline - po_order_date)
    >1 means the vendor took longer than promised; <1 means early. A PO
    missing any one of the three dates (e.g. not yet received) is excluded
    from the average, not treated as 0. A PO whose deadline equals its order
    date (zero-length promised window) is also excluded - the ratio is
    undefined, not infinite or 1.

None of these functions raise on a job with no operations, no operators, no
components, or no vendor - each independently degrades to None/0.0 exactly
as documented above, since a rollup for a freshly-created job will
legitimately have plenty of that missing.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from maxxflow_core.errors import get_logger


log = get_logger("m3_production_delay.rule_engine.elements")


def _is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def _to_days(delta: Any) -> float:
    """Both real Postgres reads (pandas Timestamp -> Timedelta) and the
    plain-datetime fake-data fixtures used in tests produce a delta with
    `.total_seconds()`; fall back to treating it as already-numeric."""
    if hasattr(delta, "total_seconds"):
        return delta.total_seconds() / 86400.0
    return float(delta)


def work_done_percentage(op: dict) -> float | None:
    """current_done_quantity / job_quantity, as a fraction (not *100). None
    when either is missing or 0 - e.g. still in a setup/prep phase before
    the first unit is finished, where there is no real progress yet to
    express as a percentage."""
    current_done = op.get("current_done_quantity")
    job_quantity = op.get("job_quantity")
    if not current_done or not job_quantity:
        return None
    return current_done / job_quantity


def time_overrun_ratio(op: dict) -> float | None:
    """None when actual_duration_minutes is missing OR 0 - a work order with
    zero actual time isn't a real "0% overrun" data point, it means no time
    has actually been logged yet (same as not having started), so it's
    excluded rather than producing a misleading ratio of 0.0. Also None when
    work_done_percentage is None - see its own docstring / this module's."""
    actual = op.get("actual_duration_minutes")
    expected = op.get("expected_duration_minutes")
    if not actual or not expected:
        return None
    pct_done = work_done_percentage(op)
    if pct_done is None:
        return None
    return actual / (pct_done * expected)


def predicted_overrun_hours(op: dict) -> float | None:
    """PROJECTS how many hours over (or under) schedule this operation is
    likely to finish. Two ways to get there, tried in order:

    1. QUANTITY-based (preferred, when there's real progress to extrapolate):
        predicted_total_duration = actual_duration_minutes * (job_quantity / current_done_quantity)
       A standard "estimate at completion" (EAC) - if X% of the quantity took
       Y minutes, finishing 100% at that same rate projects to Y/(X/100) minutes.
       This is NOT the same question as "how much of the time budget has been
       used so far" (actual - expected): an operation 60% done in 350 minutes
       against a 480-minute budget has technically used LESS time than the
       full budget so far - but at that per-unit rate, finishing the
       remaining 40% projects to ~583 total minutes, i.e. running late, not
       early. Simple actual-vs-expected can't see that until the operation is
       already overdue.

    2. OPERATOR-HISTORY fallback (when there's no quantity progress to go on
       yet - current_done_quantity is 0, e.g. still in a setup/prep phase
       before the first unit is finished, or the operation hasn't started at
       all): project using the assigned operator's own historical pace
       instead, applied to the scheduled duration:
        predicted_total_duration = expected_duration_minutes * operator_pace_ratio
       This needs no quantity data at all - just this operator's average
       elapsed/scheduled ratio from their OTHER completed work orders.

    None only when NEITHER basis is available: no quantity progress AND no
    operator pace history (e.g. a brand-new operator on an operation that
    hasn't logged any completed units yet) - there is genuinely nothing to
    project from, not a 0 or a guess.
    """
    expected = op.get("expected_duration_minutes")
    if expected is None:
        return None

    actual = op.get("actual_duration_minutes")
    pct_done = work_done_percentage(op)
    # `actual` (not `actual is not None`): a 0 here means no time has really
    # been logged yet either, same reasoning as time_overrun_ratio - fall
    # through to the operator-history basis instead of projecting from a
    # meaningless "0 minutes for this much progress" rate.
    if actual and pct_done is not None:
        predicted_total_duration = actual / pct_done
        return (predicted_total_duration - expected) / 60.0

    op_pace = operator_pace_ratio(op)
    if op_pace is None:
        return None
    predicted_total_duration = expected * op_pace
    return (predicted_total_duration - expected) / 60.0


def predecessor_time_overrun_ratio(
    op: dict, time_overrun_ratio_by_operation_id: dict[str, float | None],
) -> float | None:
    predecessor_ids = op.get("depends_on_operation_ids") or []
    if not predecessor_ids:
        return None  # independent - predecessor_time_overrun_ratio does not apply
    known_predecessor_ratios = [
        v for v in (time_overrun_ratio_by_operation_id.get(pid) for pid in predecessor_ids)
        if v is not None
    ]
    if not known_predecessor_ratios:
        return None
    return max(known_predecessor_ratios)


def _r_for_operator(operator: dict) -> float | None:
    ratios = []
    for wo in operator.get("last_10_work_orders", []):
        if _is_missing(wo.get("completed_on")):
            continue  # not yet complete - excluded from R(op), not a 0
        scheduled = wo.get("scheduled_time_minutes")
        elapsed = wo.get("elapsed_time_minutes")
        if not scheduled:
            continue
        ratios.append(elapsed / scheduled)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def operator_pace_ratio(op: dict) -> float | None:
    r_values = [
        r for r in (_r_for_operator(operator) for operator in op.get("operators", []))
        if r is not None
    ]
    if not r_values:
        return None
    return sum(r_values) / len(r_values)


def material_shortfall_ratio(op: dict) -> float:
    total = 0.0
    for component in op.get("components", []):
        required = component.get("required_quantity")
        available = component.get("available_quantity")
        if required is None or available is None:
            continue
        if available < required:
            total += math.inf if available == 0 else (required / available)
    return total


def vendor_lead_time_ratio(vendor: dict | None) -> float | None:
    if not vendor:
        return None
    ratios = []
    for po in vendor.get("last_10_purchase_orders", []):
        order_date = po.get("po_order_date")
        deadline = po.get("po_order_deadline")
        received = po.get("grn_received_date")
        if _is_missing(order_date) or _is_missing(deadline) or _is_missing(received):
            continue
        window_days = _to_days(deadline - order_date)
        if not window_days:
            continue  # zero-length promised window - ratio is undefined, not 0/inf
        elapsed_days = _to_days(received - order_date)
        ratios.append(elapsed_days / window_days)
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


#: Weighted composite risk score inputs. All four are "higher = more risk"
#: (see module docstring / composite_risk_score's own docstring for the
#: operator_pace_ratio direction note - it is NOT an ordinal skill tier).
DEFAULT_RISK_WEIGHTS: dict[str, float] = {
    "time_overrun_ratio": 0.50,
    "operator_pace_ratio": 0.35,
    "material_shortfall_ratio": 0.10,
    "supplier_reliability": 0.05,
}

#: Maps the Weight Agent's `SignalName` vocabulary (`llm_agents/weight_agent/
#: models.py`) onto this module's own risk-score keys. `seasonality` has no
#: rule-engine equivalent - elements.py computes no seasonality signal at
#: all - so it is deliberately left out of the map; `weights_bp_to_risk_weights`
#: drops it, and `composite_risk_score`'s renormalization already handles a
#: dropped signal cleanly (it renormalizes over whichever weights have a
#: matching, non-None value).
WEIGHT_AGENT_SIGNAL_TO_RISK_KEY: dict[str, str] = {
    "time_overrun": "time_overrun_ratio",
    "operator_skill": "operator_pace_ratio",
    "material_availability": "material_shortfall_ratio",
    "supplier_reliability": "supplier_reliability",
}


def weights_bp_to_risk_weights(weights_bp: dict[str, int]) -> dict[str, float]:
    """Adapts a Weight Agent `WeightResolution.weights_bp` (int basis points,
    keyed by `SignalName`, summing to 10000) into this module's own
    `risk_weights` shape (float, keyed by the rule engine's own signal names -
    see `WEIGHT_AGENT_SIGNAL_TO_RISK_KEY`), ready to pass as
    `calculate_delay_elements_for_jobs(..., risk_weights=...)`.

    Deliberately divides by 10000 (not `weight_agent.models.bp_to_percent`,
    whose own docstring says its output is "derived and lossy - never feed
    this back into internal arithmetic") to get an exact fraction of the
    whole. Any Weight Agent signal with no rule-engine equivalent (currently
    only `seasonality`) is silently dropped - there is nothing in
    `composite_risk_score`'s `values` for it to weight.
    """
    return {
        risk_key: weights_bp[signal] / 10_000.0
        for signal, risk_key in WEIGHT_AGENT_SIGNAL_TO_RISK_KEY.items()
        if signal in weights_bp
    }


def _supplier_reliability(enriched_components: list[dict]) -> float | None:
    """This operation's Supplier Reliability input: the average
    `vendor_lead_time_ratio` across whichever of its components have a vendor
    with a computable ratio. An operation with several components/vendors
    gets ONE blended number here - components with no vendor, or a vendor
    with no usable PO history, are excluded rather than counted as 0."""
    ratios = [
        c["vendor"]["vendor_lead_time_ratio"]
        for c in enriched_components
        if c.get("vendor") and c["vendor"].get("vendor_lead_time_ratio") is not None
    ]
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


#: PLACEHOLDER. There is no threshold derivable from the schema or from any
#: formula here - "how much composite risk counts as a predicted delay" is a
#: business/calibration decision, not something this code can determine on
#: its own. 1.0 is chosen only because every individual ratio equals 1.0 at
#: exactly "on schedule" (e.g. time_overrun_ratio=1.0 means actual==expected),
#: so a composite score above 1.0 means "worse than a perfectly on-time,
#: perfectly-stocked, perfectly-reliable baseline" - a defensible starting
#: point, NOT a validated cutoff. Replace this once real historical outcomes
#: (which jobs actually ended up late) are available to calibrate against -
#: the same kind of validation `maxxflow_synth/gates.py` already does for M1/M2.
DEFAULT_DELAY_THRESHOLD = 1.0


def is_delayed(
    score: float | None, threshold: float = DEFAULT_DELAY_THRESHOLD,
) -> bool | None:
    """True if `score` is strictly above `threshold`, False if at or below it,
    None if `score` itself is None (nothing to compare - not treated as
    "not delayed")."""
    if score is None:
        return None
    return score > threshold


def composite_risk_score(
    values: dict[str, float | None],
    weights: dict[str, float] = DEFAULT_RISK_WEIGHTS,
) -> float | None:
    """weight1*value1 + weight2*value2 + ... for whichever of `values` are
    not None, with the remaining weights RENORMALIZED to sum to 1 - a job
    missing operator history or vendor data still gets a score built from
    whatever signals it does have, rather than going to None outright.

    Returns None only when EVERY input is None (nothing at all to score).

    Note: `material_shortfall_ratio` can be `math.inf` (a component with zero
    available stock - see that function's docstring). This function does NOT
    cap or exclude that; a component with literally no stock at all should
    dominate the score, so an infinite shortfall correctly makes the whole
    composite_risk_score `inf` rather than being silently smoothed away.
    """
    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in weights.items():
        value = values.get(key)
        if value is None:
            continue  # excluded, not treated as 0
        weighted_sum += weight * value
        weight_total += weight

    if weight_total == 0:
        return None
    return weighted_sum / weight_total


def _calculate_delay_elements_for_one_job(
    job_rollup: dict,
    risk_weights: dict[str, float],
    delay_threshold: float,
) -> dict:
    """Returns a NEW dict shaped exactly like `job_rollup` (from
    `build_job_rollups()`), with the elements attached:
      - time_overrun_ratio, predecessor_time_overrun_ratio, operator_pace_ratio,
        material_shortfall_ratio, predicted_overrun_hours, composite_risk_score,
        is_delayed on each operation
      - vendor_lead_time_ratio on each component's `vendor` dict
    Does not mutate the input. Predecessor lookups are scoped to THIS job's
    own `operations` list only - see `calculate_delay_elements_for_jobs`.
    """
    ops = job_rollup.get("operations", [])

    # time_overrun_ratio for every operation first - predecessor_time_overrun_ratio
    # needs to look those up for arbitrary predecessors, which may appear
    # later or earlier in `ops`.
    time_overrun_ratio_by_operation_id = {
        op["operation_id"]: time_overrun_ratio(op) for op in ops
    }

    enriched_ops = []
    for op in ops:
        enriched_components = []
        for component in op.get("components", []):
            vendor = component.get("vendor")
            enriched_vendor = (
                {**vendor, "vendor_lead_time_ratio": vendor_lead_time_ratio(vendor)}
                if vendor else None
            )
            enriched_components.append({**component, "vendor": enriched_vendor})

        op_time_overrun = time_overrun_ratio_by_operation_id[op["operation_id"]]
        op_operator_pace = operator_pace_ratio(op)
        op_material_shortfall = material_shortfall_ratio(op)
        op_supplier_reliability = _supplier_reliability(enriched_components)
        risk_values = {
            "time_overrun_ratio": op_time_overrun,
            "operator_pace_ratio": op_operator_pace,
            "material_shortfall_ratio": op_material_shortfall,
            "supplier_reliability": op_supplier_reliability,
        }
        op_risk_score = composite_risk_score(risk_values, risk_weights)
        op_predicted_overrun = predicted_overrun_hours(op)
        active_terms = {
            key: {
                "value": risk_values.get(key),
                "weight": weight,
                "weighted_value": weight * risk_values[key],
            }
            for key, weight in risk_weights.items()
            if risk_values.get(key) is not None
        }
        active_weight_total = sum(term["weight"] for term in active_terms.values())
        weighted_sum = sum(term["weighted_value"] for term in active_terms.values())
        delayed = is_delayed(op_risk_score, delay_threshold)

        log.info(
            "m3_risk_calculation job_id=%s operation_id=%s operation_name=%s "
            "expected_minutes=%s actual_minutes=%s job_quantity=%s done_quantity=%s "
            "predicted_overrun_hours=%s signals=%s active_terms=%s weighted_sum=%s "
            "active_weight_total=%s risk_score=%s delay_threshold=%s is_delayed=%s",
            job_rollup.get("job_id"),
            op.get("operation_id"),
            op.get("operation_name"),
            op.get("expected_duration_minutes"),
            op.get("actual_duration_minutes"),
            op.get("job_quantity"),
            op.get("current_done_quantity"),
            op_predicted_overrun,
            risk_values,
            active_terms,
            weighted_sum,
            active_weight_total,
            op_risk_score,
            delay_threshold,
            delayed,
        )

        enriched_ops.append({
            **op,
            "components": enriched_components,
            "time_overrun_ratio": op_time_overrun,
            "predicted_overrun_hours": op_predicted_overrun,
            "predecessor_time_overrun_ratio": predecessor_time_overrun_ratio(
                op, time_overrun_ratio_by_operation_id,
            ),
            "operator_pace_ratio": op_operator_pace,
            "material_shortfall_ratio": op_material_shortfall,
            "composite_risk_score": op_risk_score,
            "is_delayed": delayed,
        })

    return {**job_rollup, "operations": enriched_ops}


def calculate_delay_elements_for_jobs(
    job_rollups: list[dict],
    risk_weights: dict[str, float] = DEFAULT_RISK_WEIGHTS,
    delay_threshold: float = DEFAULT_DELAY_THRESHOLD,
) -> list[dict]:
    """Computes every delay element for one or more jobs (e.g. the full list
    `build_job_rollups()` returns for a whole tenant) - pass a single-item
    list (`[one_job]`) for one job, there is no separate single-job function,
    this covers both.

    Each job is processed independently: one job's operations, operators,
    components and predecessors never affect another job's results - a
    `depends_on_operation_ids` entry that happens to name an operation id
    from a DIFFERENT job resolves to nothing, never that other job's data
    (see test_calculate_delay_elements_for_jobs_processes_every_job).
    """
    return [
        _calculate_delay_elements_for_one_job(job, risk_weights, delay_threshold)
        for job in job_rollups
    ]
