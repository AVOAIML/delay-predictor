"""Turns ONE scored job — exactly the dict
``rule_engine.elements.calculate_delay_elements_for_jobs()`` returns — into an
:class:`~m3_production_delay.review.schemas.EvidencePack`.

This is the only module that reads the rule engine's output shape, and it
computes nothing the rule engine already computed. What it does add is the
four things Section 3 needs that the engine does not expose, each derived
here and nowhere else:

  * **supplier_reliability** — the engine computes it (privately) and feeds it
    into ``composite_risk_score``, but does not attach it to the operation.
    Recomputed here from the vendor ratios the engine DOES expose, by the same
    definition: the mean of ``components[].vendor.vendor_lead_time_ratio``
    over whichever components have one.
  * **overrun_basis** — which of the two projections behind
    ``predicted_overrun_hours`` was used. Stated so a summary can never imply
    a quantity-extrapolated projection when the number actually came from an
    operator's historical pace.
  * **contribution** — a signal's share of the composite score, corrected for
    the renormalisation ``composite_risk_score`` performs over whichever
    weights had a non-None value. Shares over one operation's signals sum to
    1, which is what makes "ordered by contribution" meaningful even when the
    tenant's weight vector does not sum to 1 (it never does — the Weight
    Agent's ``seasonality`` share has no rule-engine equivalent and is
    dropped).
  * **is_scorable** — the user story's 25%-progress gate. The engine scores
    every operation regardless of progress, so an operation that has not
    started can still come back ``is_delayed=True`` on operator, material and
    supplier signals alone. Section 3 will not explain such an operation.

Two structural facts about the rollup shape drive the rest of the design.
Components are attached per *manufacturing order*, not per operation, so
every operation of a job carries the identical component list and therefore
the identical material and supplier signals — those are de-duplicated into
job-scoped evidence here (:data:`~schemas.JOB_SCOPED_SIGNALS`). And there is
no job-level roll-up at all, so the summary is computed here, over scorable
operations only, as an explicit worst-operation aggregation.
"""

from __future__ import annotations

import math
from typing import Any

from maxxflow_core.errors import get_logger

from m3_production_delay.review.schemas import (
    BASIS_NONE,
    BASIS_OPERATOR_PACE,
    BASIS_QUANTITY,
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_PREDECESSOR_OVERRUN,
    SIGNAL_SUPPLIER_RELIABILITY,
    SIGNAL_TIME_OVERRUN,
    SUMMARY_BASIS_WORST_OPERATION,
    ComponentEvidence,
    EvidencePack,
    Issue,
    OperationEvidence,
    SignalEvidence,
    finite_or_none,
)

log = get_logger("m3_production_delay.review.evidence")

#: THE fire baselines. Every "did this signal fire" decision in Section 3
#: reads this table — the composer, the validators and the judge prompt all
#: derive from it rather than restating a threshold of their own, so the
#: three can never drift apart. Comparison is strictly greater-than in every
#: case, which is what makes ``material_shortfall_ratio > 0`` mean "at least
#: one component is genuinely short" (the engine's own value is exactly 0.0
#: when nothing is short).
FIRE_BASELINES: dict[str, float] = {
    SIGNAL_TIME_OVERRUN: 1.0,
    SIGNAL_OPERATOR_PACE: 1.2,
    SIGNAL_MATERIAL_SHORTFALL: 0.0,
    SIGNAL_SUPPLIER_RELIABILITY: 1.0,
    SIGNAL_PREDECESSOR_OVERRUN: 1.2,
}

#: The signals ``composite_risk_score`` actually weights, in the order the
#: rule engine's own ``DEFAULT_RISK_WEIGHTS`` lists them. ``supplier_reliability``
#: is included: it IS weighted, it is simply not attached to the operation.
WEIGHTED_SIGNALS: tuple[str, ...] = (
    SIGNAL_TIME_OVERRUN,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_SUPPLIER_RELIABILITY,
)

#: Minimum elapsed share of the planned duration before this agent will
#: explain an operation (user story: "25% progress"). Measured on
#: ``time_overrun_ratio`` = actual/expected, which is the only progress
#: measure available for an operation with no finished units yet.
SCORING_GATE_MIN_TIME_RATIO = 0.25


def fired(key: str, value: float | None) -> bool:
    """Single fire decision for every caller in Section 3."""
    if value is None:
        return False
    return value > FIRE_BASELINES[key]


def supplier_reliability(components: list[dict] | tuple[dict, ...]) -> float | None:
    """This operation's Supplier Reliability input, recomputed from the vendor
    ratios the rule engine attaches to each component.

    Same definition the engine uses internally: the mean over whichever
    components have a vendor with a computable ``vendor_lead_time_ratio``.
    A component with no vendor, or a vendor with no usable PO history, is
    excluded rather than counted as an on-time 1.0 — the same reason a
    missing signal is excluded from the composite rather than zero-filled.
    """
    ratios = [
        ratio
        for ratio in (
            finite_or_none((component.get("vendor") or {}).get("vendor_lead_time_ratio"))
            for component in components
        )
        if ratio is not None
    ]
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def overrun_basis(op: dict) -> str:
    """Which projection produced ``predicted_overrun_hours``.

    Mirrors the branch in ``elements.predicted_overrun_hours``: a quantity
    extrapolation when there is real progress to extrapolate from, the
    assigned operator's historical pace when there is not, and nothing at all
    when neither is available.
    """
    if (
        op.get("actual_duration_minutes") is not None
        and op.get("current_done_quantity")
        and op.get("job_quantity")
    ):
        return BASIS_QUANTITY
    if finite_or_none(op.get("operator_pace_ratio")) is not None:
        return BASIS_OPERATOR_PACE
    return BASIS_NONE


def is_scorable(op: dict) -> bool:
    """The user story's scoring gate, which the rule engine does not apply.

    Requires both that time has actually been logged and that at least 25% of
    the planned duration has elapsed. An operation below that has too little
    signal to attribute a cause to, however high its composite score climbs
    on operator/material/supplier history alone.
    """
    if op.get("actual_duration_minutes") is None:
        return False
    ratio = finite_or_none(op.get("time_overrun_ratio"))
    if ratio is None:
        return False
    return ratio >= SCORING_GATE_MIN_TIME_RATIO


def operator_history_depth(op: dict) -> int:
    """How many completed work orders the operator pace was averaged over.

    Mirrors ``elements._r_for_operator``'s own filter — a work order counts
    only once it has completed and has a scheduled duration to compare
    against. Carried into the evidence because the line built on this signal
    says "over recent completed jobs", and a judge reading a bare
    ``pace_ratio`` has no way to tell whether any such jobs exist. A claim the
    evidence cannot substantiate is one this pipeline should not be making.
    """
    completed = 0
    for operator in op.get("operators") or []:
        for work_order in operator.get("last_10_work_orders") or []:
            if work_order.get("completed_on") is None:
                continue
            if not work_order.get("scheduled_time_minutes"):
                continue
            completed += 1
    return completed


def operation_label(op: dict) -> str:
    """A human-readable subject for a line.

    Uses a real name only when the rollup carries one. It does not resolve a
    name from anywhere else and never invents one: an operation with no name
    is identified by its master-data type code and a short id, which is
    unambiguous on screen and honest about what is known.
    """
    for key in ("operation_name", "work_center_name"):
        name = op.get(key)
        if name:
            return str(name)
    operation_id = str(op.get("operation_id") or "")
    operation_type = op.get("operation_type") or "Operation"
    return f"{operation_type} · {operation_id[:8]}" if operation_id else str(operation_type)


def _coerce(
    key: str, raw: Any, *, ref: str, issues: list[Issue], label: str | None = None
) -> float | None:
    """``finite_or_none`` plus the Issue that makes a coercion visible.

    A ``math.inf`` here is not a glitch: ``material_shortfall_ratio`` is
    infinite by design whenever a short component has zero stock, and that
    infinity propagates into ``composite_risk_score``. Dropping it silently
    would leave the panel showing a blank where the strongest possible
    shortfall signal was.
    """
    value = finite_or_none(raw)
    if value is None and raw is not None:
        issues.append(
            Issue(
                check="non_finite",
                severity="warning",
                message=(
                    f"{label or key} is not a finite number ({raw!r}) and was dropped from the "
                    "insight; a zero-stock component makes the shortfall ratio infinite"
                ),
                ref=ref,
            )
        )
    return value


def _short_components(op: dict, issues: list[Issue]) -> list[ComponentEvidence]:
    """Components of this operation whose available stock is below what the
    job requires, carrying the shortfall QUANTITY — the number a planner can
    act on — rather than the ratio that went into the score."""
    short: list[ComponentEvidence] = []
    for component in op.get("components") or []:
        required = finite_or_none(component.get("required_quantity"))
        available = finite_or_none(component.get("available_quantity"))
        if required is None or available is None or available >= required:
            continue
        vendor = component.get("vendor") or {}
        component_id = str(component.get("component_id") or "")
        if not component_id:
            issues.append(
                Issue(
                    check="missing_component_id",
                    severity="warning",
                    message="a short component has no component_id and was dropped",
                    ref=str(op.get("operation_id") or ""),
                )
            )
            continue
        short.append(
            ComponentEvidence(
                component_id=component_id,
                name=component.get("name"),
                required_quantity=required,
                available_quantity=available,
                shortfall_quantity=required - available,
                vendor_name=vendor.get("name"),
                vendor_lead_time_ratio=finite_or_none(vendor.get("vendor_lead_time_ratio")),
            )
        )
    return short


def _active_weight_total(values: dict[str, float | None], weights: dict[str, float]) -> float:
    """The denominator ``composite_risk_score`` renormalised by: the sum of
    the weights whose signal actually had a value. Reproduced here rather
    than assumed to be 1.0, because a tenant's resolved weights never sum to
    1 and a job with missing operator/vendor history drops more still."""
    return sum(
        weight for key, weight in weights.items() if values.get(key) is not None and weight > 0
    )


def _contribution(
    value: float | None, weight: float, score: float | None, weight_total: float
) -> float | None:
    """This signal's share of the composite score, or None when the share is
    not computable at all.

    ``weight * value / (score * Σweight_active)`` — the ``Σweight_active``
    factor undoes the composite's renormalisation, so shares over one
    operation's signals sum to 1 rather than to the weight vector's total.
    """
    if value is None or score is None or weight <= 0 or weight_total <= 0:
        return None
    denominator = score * weight_total
    if denominator == 0:
        return None
    share = weight * value / denominator
    return share if math.isfinite(share) else None


def _signal(
    key: str,
    value: float | None,
    weights: dict[str, float],
    score: float | None,
    weight_total: float,
    detail: dict[str, Any],
) -> SignalEvidence:
    weight = float(weights.get(key, 0.0) or 0.0)
    return SignalEvidence(
        key=key,
        value=value,
        weight=weight,
        fired=fired(key, value),
        contribution=_contribution(value, weight, score, weight_total),
        detail=detail,
    )


def _operation_evidence(
    op: dict, weights: dict[str, float], issues: list[Issue]
) -> OperationEvidence:
    operation_id = str(op.get("operation_id") or "")
    if not operation_id:
        raise ValueError("every operation in a scored job must carry an operation_id")
    if "composite_risk_score" not in op:
        raise ValueError(
            f"operation {operation_id!r} has no composite_risk_score — build_evidence() takes the "
            "output of calculate_delay_elements_for_jobs(), not a raw build_job_rollups() job"
        )

    label = operation_label(op)
    time_overrun = _coerce(
        SIGNAL_TIME_OVERRUN, op.get("time_overrun_ratio"), ref=operation_id, issues=issues,
        label=f"{label} time_overrun_ratio",
    )
    operator_pace = _coerce(
        SIGNAL_OPERATOR_PACE, op.get("operator_pace_ratio"), ref=operation_id, issues=issues,
        label=f"{label} operator_pace_ratio",
    )
    material = _coerce(
        SIGNAL_MATERIAL_SHORTFALL, op.get("material_shortfall_ratio"), ref=operation_id,
        issues=issues, label=f"{label} material_shortfall_ratio",
    )
    predecessor = _coerce(
        SIGNAL_PREDECESSOR_OVERRUN, op.get("predecessor_time_overrun_ratio"), ref=operation_id,
        issues=issues, label=f"{label} predecessor_time_overrun_ratio",
    )
    components = op.get("components") or []
    supplier = _coerce(
        SIGNAL_SUPPLIER_RELIABILITY, supplier_reliability(components), ref=operation_id,
        issues=issues, label=f"{label} supplier_reliability",
    )
    score = _coerce(
        "composite_risk_score", op.get("composite_risk_score"), ref=operation_id, issues=issues,
        label=f"{label} composite_risk_score",
    )

    expected = finite_or_none(op.get("expected_duration_minutes"))
    actual = finite_or_none(op.get("actual_duration_minutes"))
    short_components = _short_components(op, issues)

    # Weighted values only — the composite never saw the predecessor signal,
    # so it must not appear in the renormalisation denominator either.
    weighted_values: dict[str, float | None] = {
        SIGNAL_TIME_OVERRUN: time_overrun,
        SIGNAL_OPERATOR_PACE: operator_pace,
        SIGNAL_MATERIAL_SHORTFALL: material,
        SIGNAL_SUPPLIER_RELIABILITY: supplier,
    }
    weight_total = _active_weight_total(weighted_values, weights)

    # Exactly the vendors whose ratios the supplier mean was taken over, so a
    # line naming them is quoting the same set the number came from.
    vendor_names = sorted(
        {
            str((component.get("vendor") or {}).get("name"))
            for component in components
            if (component.get("vendor") or {}).get("name")
            and finite_or_none((component.get("vendor") or {}).get("vendor_lead_time_ratio"))
            is not None
        }
    )

    details: dict[str, dict[str, Any]] = {
        SIGNAL_TIME_OVERRUN: {
            "actual_minutes": actual,
            "expected_minutes": expected,
            "actual_hrs": None if actual is None else actual / 60.0,
            "expected_hrs": None if expected is None else expected / 60.0,
            "delta_hrs": (
                None if actual is None or expected is None else (actual - expected) / 60.0
            ),
        },
        SIGNAL_OPERATOR_PACE: {
            "pace_ratio": operator_pace,
            "operator_count": op.get("operator_count"),
            # What "over recent completed jobs" actually rests on. Without it
            # the pace is an unsourced number and the line's wording is a claim
            # the evidence cannot back.
            "completed_work_orders": operator_history_depth(op),
        },
        SIGNAL_MATERIAL_SHORTFALL: {
            "short_components": [c.to_dict() for c in short_components],
        },
        SIGNAL_SUPPLIER_RELIABILITY: {
            "vendor_names": [name for name in vendor_names if name],
            "mean_lead_time_ratio": supplier,
        },
        SIGNAL_PREDECESSOR_OVERRUN: {
            "ratio": predecessor,
            "predecessor_operation_ids": list(op.get("depends_on_operation_ids") or []),
        },
    }

    signals = tuple(
        _signal(key, value, weights, score, weight_total, details[key])
        for key, value in (
            (SIGNAL_TIME_OVERRUN, time_overrun),
            (SIGNAL_OPERATOR_PACE, operator_pace),
            (SIGNAL_MATERIAL_SHORTFALL, material),
            (SIGNAL_SUPPLIER_RELIABILITY, supplier),
        )
    ) + (
        # Computed by the engine but never an input to composite_risk_score.
        # Its weight is pinned to 0 here rather than read from the tenant
        # vector, so a stray key in a weight dict can never promote a
        # cascading-delay flag into a weighted cause with a contribution.
        SignalEvidence(
            key=SIGNAL_PREDECESSOR_OVERRUN,
            value=predecessor,
            weight=0.0,
            fired=fired(SIGNAL_PREDECESSOR_OVERRUN, predecessor),
            contribution=None,
            detail=details[SIGNAL_PREDECESSOR_OVERRUN],
        ),
    )

    return OperationEvidence(
        operation_id=operation_id,
        operation_name=label,
        status=op.get("status"),
        expected_duration_minutes=expected,
        actual_duration_minutes=actual,
        job_quantity=finite_or_none(op.get("job_quantity")),
        current_done_quantity=finite_or_none(op.get("current_done_quantity")),
        composite_risk_score=score,
        # bool(), not the raw value: the engine returns a plain bool today,
        # but a numpy bool_ arriving from a DataFrame-backed rollup would
        # fail OperationEvidence's type guard for no useful reason.
        is_delayed=None if op.get("is_delayed") is None else bool(op["is_delayed"]),
        predicted_overrun_hours=_coerce(
            "predicted_overrun_hours", op.get("predicted_overrun_hours"), ref=operation_id,
            issues=issues, label=f"{label} predicted_overrun_hours",
        ),
        overrun_basis=overrun_basis(op),
        signals=signals,
        components_short=tuple(short_components),
        depends_on_operation_ids=tuple(
            str(pid) for pid in (op.get("depends_on_operation_ids") or [])
        ),
        is_scorable=is_scorable(op),
    )


def _material_overrun(operations: tuple[OperationEvidence, ...]) -> tuple[ComponentEvidence, ...]:
    """The job's short components, each once.

    ``rollup.py`` attaches the manufacturing order's whole component list to
    every one of its operations, so the same shortage appears N times in a
    job with N operations. First occurrence wins; the component rows are
    identical by construction, so which one wins cannot change a number.
    """
    seen: dict[str, ComponentEvidence] = {}
    for op in operations:
        for component in op.components_short:
            seen.setdefault(component.component_id, component)
    return tuple(seen.values())


def _job_signals(
    operations: tuple[OperationEvidence, ...],
    material_overrun: tuple[ComponentEvidence, ...],
    weights: dict[str, float],
    issues: list[Issue],
) -> tuple[SignalEvidence, ...]:
    """Material and supplier evidence at the job level, built from the
    de-duplicated component set.

    The value is recomputed from the de-duplicated components (identical to
    any single operation's, since the component list is MO-wide). The
    contribution is the share the signal held on the operation the SUMMARY
    comes from — the worst scorable one. A job-scoped line has to sort
    against operation-scoped ones somehow, and the summary operation is the
    only denominator that makes those shares comparable: taking the largest
    share across all operations instead would let a low-scoring operation
    inflate a job-wide line to the top of the list, since the same shortfall
    is a bigger fraction of a smaller score.
    """
    shortfall_total = 0.0
    for component in material_overrun:
        required = component.required_quantity
        available = component.available_quantity
        if required is None or available is None:
            continue
        shortfall_total += math.inf if available == 0 else required / available
    material_value = _coerce(
        SIGNAL_MATERIAL_SHORTFALL, shortfall_total, ref="job", issues=issues,
        label="job material_shortfall_ratio",
    )

    # The supplier mean is taken from an operation's own evidence rather than
    # recomputed over the short components: the engine averages across EVERY
    # component that has a vendor ratio, not only the short ones, and the
    # job-level number has to be the one that actually moved the score. Value
    # and vendor names are read from the same operation in one step, so a line
    # can never name a set of vendors the quoted mean was not taken over.
    supplier_value: float | None = None
    supplier_detail: dict[str, Any] = {}
    for op in operations:
        signal = op.signal(SIGNAL_SUPPLIER_RELIABILITY)
        if signal is not None and signal.value is not None:
            supplier_value = signal.value
            supplier_detail = dict(signal.detail)
            break

    summary_op = _summary_operation(operations)

    def best_contribution(key: str) -> float | None:
        if summary_op is not None:
            signal = summary_op.signal(key)
            if signal is not None and signal.contribution is not None:
                return signal.contribution
        shares = [
            signal.contribution
            for op in operations
            if op.is_scorable
            for signal in op.signals
            if signal.key == key and signal.contribution is not None
        ]
        return max(shares) if shares else None

    material_detail: dict[str, Any] = {
        "short_components": [c.to_dict() for c in material_overrun]
    }
    supplier_detail["mean_lead_time_ratio"] = supplier_value

    return (
        SignalEvidence(
            key=SIGNAL_MATERIAL_SHORTFALL,
            value=material_value,
            weight=float(weights.get(SIGNAL_MATERIAL_SHORTFALL, 0.0) or 0.0),
            fired=fired(SIGNAL_MATERIAL_SHORTFALL, material_value),
            contribution=best_contribution(SIGNAL_MATERIAL_SHORTFALL),
            detail=material_detail,
        ),
        SignalEvidence(
            key=SIGNAL_SUPPLIER_RELIABILITY,
            value=supplier_value,
            weight=float(weights.get(SIGNAL_SUPPLIER_RELIABILITY, 0.0) or 0.0),
            fired=fired(SIGNAL_SUPPLIER_RELIABILITY, supplier_value),
            contribution=best_contribution(SIGNAL_SUPPLIER_RELIABILITY),
            detail=supplier_detail,
        ),
    )


def _assert_threshold_matches(
    job_id: str, operations: tuple[OperationEvidence, ...], threshold: float
) -> None:
    """The threshold handed in must be the one the Risk Engine scored with.

    ``is_delayed`` is the engine's own verdict, baked in when the job was
    scored; the summary and the validators compare it against ``threshold``.
    Review a job at a different cutoff and every badge in the payload is
    attributed to a number that did not produce it — so this is a caller
    error, caught here with a message naming the cause, rather than a
    downstream validator failure whose message would only describe the
    symptom.
    """
    for op in operations:
        score = op.composite_risk_score
        if score is None or op.is_delayed is None:
            continue  # nothing to compare — see the finiteness coercion above
        if op.is_delayed != (score > threshold):
            raise ValueError(
                f"job {job_id!r} operation {op.operation_id!r} was scored is_delayed="
                f"{op.is_delayed} at composite_risk_score={score}, which contradicts "
                f"threshold={threshold}. Pass the same delay_threshold that "
                "calculate_delay_elements_for_jobs() was called with — reviewing a job at a "
                "different cutoff would attribute its badge to a number that did not produce it."
            )


def _summary_operation(
    operations: tuple[OperationEvidence, ...],
) -> OperationEvidence | None:
    """The scorable operation the job summary comes from: the worst one."""
    candidates = [
        op for op in operations if op.is_scorable and op.composite_risk_score is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda op: op.composite_risk_score)


def _summarise(
    operations: tuple[OperationEvidence, ...],
) -> tuple[float | None, float | None, bool | None]:
    """Job-level summary over SCORABLE operations only — the rule engine has
    no roll-up of its own.

    Worst operation, not an average and not a sum: a job is late when its
    worst step is late, and summing projected overruns across operations
    would double-count work that runs in parallel. ``is_delayed`` is
    aggregated only over operations whose composite score survived the
    finiteness coercion, which keeps ``summary_is_delayed`` exactly equal to
    ``summary_risk_score > threshold``.
    """
    scorable = [op for op in operations if op.is_scorable]
    scores = [op.composite_risk_score for op in scorable if op.composite_risk_score is not None]
    hours = [op.predicted_overrun_hours for op in scorable if op.predicted_overrun_hours is not None]
    flags = [
        op.is_delayed
        for op in scorable
        if op.is_delayed is not None and op.composite_risk_score is not None
    ]
    return (
        max(scores) if scores else None,
        max(hours) if hours else None,
        any(flags) if flags else None,
    )


def build_evidence(job: dict, weights: dict[str, float], threshold: float) -> EvidencePack:
    """Build the evidence pack for one scored job.

    ``job`` is one element of ``calculate_delay_elements_for_jobs()``'s
    result. ``weights`` is the same vector that produced its scores — the
    tenant's resolved ``risk_weights``, whose keys are the rule engine's own
    signal names. ``threshold`` has no default anywhere in Section 3: the
    same composite score means "delayed" or "fine" depending on the tenant,
    so it is always an explicit argument.
    """
    job_id = str(job.get("job_id") or "")
    if not job_id:
        raise ValueError("a scored job must carry a job_id")
    if threshold is None:
        raise ValueError("threshold is required — Section 3 has no default delay threshold")

    issues: list[Issue] = []
    operations = tuple(
        _operation_evidence(op, weights, issues) for op in (job.get("operations") or [])
    )
    _assert_threshold_matches(job_id, operations, float(threshold))
    material_overrun = _material_overrun(operations)
    job_signals = _job_signals(operations, material_overrun, weights, issues)
    summary_score, summary_hours, summary_delayed = _summarise(operations)

    pack = EvidencePack(
        job_id=job_id,
        delay_threshold=float(threshold),
        weights={key: float(value) for key, value in (weights or {}).items()},
        operations=operations,
        material_overrun=material_overrun,
        job_signals=job_signals,
        summary_risk_score=summary_score,
        summary_overrun_hours=summary_hours,
        summary_is_delayed=summary_delayed,
        summary_basis=SUMMARY_BASIS_WORST_OPERATION,
        issues=tuple(issues),
    )
    log.debug(
        "m3_review evidence job_id=%s operations=%d scorable=%d short_components=%d issues=%d",
        pack.job_id,
        len(pack.operations),
        len(pack.scorable_operations),
        len(pack.material_overrun),
        len(pack.issues),
    )
    return pack
