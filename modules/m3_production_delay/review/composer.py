"""Turns an :class:`~m3_production_delay.review.schemas.EvidencePack` into the
candidate explanation — :class:`~m3_production_delay.review.schemas.InsightDraft`.

Every line this module emits is a template filled from evidence fields. There
is no LLM here and no path for one: the judge downstream can approve a line or
cause it to be dropped, but it can never write, reword or reorder one. That is
what makes the "no number is ever produced by a language model" property
structural rather than a prompt instruction.

Three rules shape what gets emitted:

  * **One line per fired score cause.** A signal that did not clear its
    fire baseline (``evidence.FIRE_BASELINES``) is not a cause, and a signal
    the tenant weights at zero did not move the score — neither earns a
    sentence. A validated ``critical_path_cascade_ratio`` is the exception to
    tenant weighting: it is a deterministic score overlay and earns its own
    line. The raw ``predecessor_time_overrun_ratio`` remains context only.
  * **Ordered by contribution.** The biggest share of the composite score
    reads first. Ties break on the canonical signal order, so the same
    evidence always composes the same list in the same sequence.
  * **Material and supplier are job-scoped.** Their evidence is MO-wide, so
    they are stated once for the job rather than repeated identically under
    every operation.

``material_overrun`` rides on the draft independently of the score: a short
component is a fact about the warehouse worth showing whether or not the
composite crossed the delay threshold.

Every numeral rendered into a line is also recorded in that line's ``quoted``
map, which is what lets ``validators.check_quoted_numbers`` re-derive the
prose from the evidence instead of trusting it. A template that renders a
number without quoting it will fail that validator — deliberately.
"""

from __future__ import annotations

from maxxflow_core.errors import get_logger

from m3_production_delay.review.schemas import (
    SCOPE_JOB,
    SCOPE_OPERATION,
    SIGNAL_CRITICAL_PATH_CASCADE,
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_ORDER,
    SIGNAL_SUPPLIER_RELIABILITY,
    SIGNAL_TIME_OVERRUN,
    ComponentEvidence,
    EvidencePack,
    InsightDraft,
    InsightLine,
    OperationEvidence,
    SignalEvidence,
)

log = get_logger("m3_production_delay.review.composer")

#: The user story's headlines, verbatim. Kept in one table so the wording is
#: reviewable in one place and so ``validators.check_forbidden_causes`` is
#: checking the same strings the composer can actually produce.
HEADLINES: dict[str, str] = {
    SIGNAL_TIME_OVERRUN: "Actual time logged has exceeded expected duration",
    SIGNAL_OPERATOR_PACE: "Assigned operator has a history of overrunning",
    SIGNAL_MATERIAL_SHORTFALL: "Required material is short in the warehouse",
    SIGNAL_SUPPLIER_RELIABILITY: "Supplier for a short component has a late-delivery history",
    SIGNAL_CRITICAL_PATH_CASCADE: "A critical-path predecessor is overrunning",
}

#: How many short components a single material line names before summarising
#: the rest as "+N more". The remaining count is itself quoted, so it stays
#: traceable to the evidence like every other numeral.
MAX_NAMED_COMPONENTS = 3

#: Display precision. The validators' tolerance floor is set to half of the
#: coarsest of these (0.05, for one-decimal hours) so correct rounding can
#: never be mistaken for a misquoted number.
_HOURS_DP = 1
_RATIO_DP = 2


def _hrs(value: float) -> str:
    return f"{value:.{_HOURS_DP}f}"


def _ratio(value: float) -> str:
    return f"{value:.{_RATIO_DP}f}"


def _qty(value: float) -> str:
    """Quantities read as whole units when they are whole — a shortfall of 60
    pieces should not render as "60.00 short"."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _time_line(op: OperationEvidence, signal: SignalEvidence) -> InsightLine | None:
    detail = signal.detail
    actual_hrs = detail.get("actual_hrs")
    expected_hrs = detail.get("expected_hrs")
    delta_hrs = detail.get("delta_hrs")
    if actual_hrs is None or expected_hrs is None or delta_hrs is None:
        # The signal fired, so a ratio existed; without both durations there
        # is no sentence that can be written without inventing a number.
        return None
    return InsightLine(
        index=0,
        signal_key=SIGNAL_TIME_OVERRUN,
        scope=SCOPE_OPERATION,
        headline=HEADLINES[SIGNAL_TIME_OVERRUN],
        detail=f"{op.operation_name} · {_hrs(actual_hrs)} hrs logged of {_hrs(expected_hrs)} planned",
        delta=f"+{_hrs(delta_hrs)} hrs",
        operation_id=op.operation_id,
        contribution=signal.contribution,
        quoted={
            "actual_hrs": actual_hrs,
            "expected_hrs": expected_hrs,
            "delta_hrs": delta_hrs,
        },
    )


def _operator_line(op: OperationEvidence, signal: SignalEvidence) -> InsightLine | None:
    pace = signal.value
    if pace is None:
        return None
    return InsightLine(
        index=0,
        signal_key=SIGNAL_OPERATOR_PACE,
        scope=SCOPE_OPERATION,
        headline=HEADLINES[SIGNAL_OPERATOR_PACE],
        detail=(
            f"{op.operation_name} · operator pace {_ratio(pace)}× planned over recent "
            "completed jobs"
        ),
        operation_id=op.operation_id,
        contribution=signal.contribution,
        quoted={"pace_ratio": pace},
    )


def _cascade_line(op: OperationEvidence, signal: SignalEvidence) -> InsightLine | None:
    ratio = signal.value
    if ratio is None:
        return None
    return InsightLine(
        index=0,
        signal_key=SIGNAL_CRITICAL_PATH_CASCADE,
        scope=SCOPE_OPERATION,
        headline=HEADLINES[SIGNAL_CRITICAL_PATH_CASCADE],
        detail=(
            f"{op.operation_name} · inherited critical-path overrun {_ratio(ratio)}× planned"
        ),
        operation_id=op.operation_id,
        contribution=signal.contribution,
        quoted={"cascade_ratio": ratio},
    )


def _material_line(
    job_id: str, signal: SignalEvidence, components: tuple[ComponentEvidence, ...]
) -> InsightLine | None:
    named = [c for c in components if c.shortfall_quantity is not None][:MAX_NAMED_COMPONENTS]
    if not named:
        return None
    quoted: dict[str, float] = {}
    parts: list[str] = []
    for component in named:
        label = component.name or component.component_id
        parts.append(f"{label} – {_qty(component.shortfall_quantity)} short")
        quoted[f"shortfall:{component.component_id}"] = component.shortfall_quantity
    remaining = len(components) - len(named)
    if remaining > 0:
        parts.append(f"+{remaining} more")
        quoted["more_count"] = float(remaining)
    return InsightLine(
        index=0,
        signal_key=SIGNAL_MATERIAL_SHORTFALL,
        scope=SCOPE_JOB,
        headline=HEADLINES[SIGNAL_MATERIAL_SHORTFALL],
        detail=f"{job_id} · " + "; ".join(parts),
        contribution=signal.contribution,
        quoted=quoted,
    )


def _supplier_line(job_id: str, signal: SignalEvidence) -> InsightLine | None:
    ratio = signal.value
    vendor_names = [str(name) for name in (signal.detail.get("vendor_names") or []) if name]
    if ratio is None or not vendor_names:
        # Without the vendors the mean was taken over there is nothing to
        # attribute the lateness to, and "a supplier" is not a claim worth
        # putting on screen.
        return None
    return InsightLine(
        index=0,
        signal_key=SIGNAL_SUPPLIER_RELIABILITY,
        scope=SCOPE_JOB,
        headline=HEADLINES[SIGNAL_SUPPLIER_RELIABILITY],
        detail=(
            f"{job_id} · vendors {', '.join(vendor_names)} average lead-time ratio {_ratio(ratio)}"
        ),
        contribution=signal.contribution,
        quoted={"mean_lead_time_ratio": ratio},
    )


def _emittable(signal: SignalEvidence | None) -> bool:
    """A signal earns a line only if it fired and changed the score.

    Usually that means it carried tenant weight. A critical-path cascade can
    also qualify through its explicit deterministic score contribution.
    """
    return signal is not None and signal.fired and signal.affects_score


def _sort_key(line: InsightLine, operation_order: dict[str, int]) -> tuple:
    """Contribution descending, then canonical signal order, then the
    operation's position in the job — a total order, so composing the same
    evidence twice always produces the same list."""
    return (
        -(line.contribution if line.contribution is not None else -1.0),
        SIGNAL_ORDER.index(line.signal_key),
        operation_order.get(line.operation_id or "", -1),
    )


def compose(pack: EvidencePack) -> InsightDraft:
    """Compose the candidate insight for one job.

    A job with no scorable operation composes no lines at all: the summary
    numbers exist but nothing in the evidence is complete enough to attribute
    a cause to, and the pipeline suppresses such a job rather than publishing
    an unexplained badge.
    """
    scorable = pack.scorable_operations
    lines: list[InsightLine] = []

    if scorable:
        for op in scorable:
            time_signal = op.signal(SIGNAL_TIME_OVERRUN)
            if _emittable(time_signal):
                line = _time_line(op, time_signal)
                if line is not None:
                    lines.append(line)
            operator_signal = op.signal(SIGNAL_OPERATOR_PACE)
            if _emittable(operator_signal):
                line = _operator_line(op, operator_signal)
                if line is not None:
                    lines.append(line)
            cascade_signal = op.signal(SIGNAL_CRITICAL_PATH_CASCADE)
            if _emittable(cascade_signal):
                line = _cascade_line(op, cascade_signal)
                if line is not None:
                    lines.append(line)

        material_signal = pack.job_signal(SIGNAL_MATERIAL_SHORTFALL)
        if _emittable(material_signal):
            line = _material_line(pack.job_id, material_signal, pack.material_overrun)
            if line is not None:
                lines.append(line)
        supplier_signal = pack.job_signal(SIGNAL_SUPPLIER_RELIABILITY)
        if _emittable(supplier_signal):
            line = _supplier_line(pack.job_id, supplier_signal)
            if line is not None:
                lines.append(line)

    operation_order = {op.operation_id: i for i, op in enumerate(pack.operations)}
    lines.sort(key=lambda line: _sort_key(line, operation_order))
    ordered = tuple(line.renumbered(i) for i, line in enumerate(lines))

    draft = InsightDraft(
        job_id=pack.job_id,
        summary_overrun_hours=pack.summary_overrun_hours,
        summary_risk_score=pack.summary_risk_score,
        summary_is_delayed=pack.summary_is_delayed,
        summary_basis=pack.summary_basis,
        why_lines=ordered,
        material_overrun=pack.material_overrun,
    )
    log.debug(
        "m3_review composed job_id=%s lines=%d scorable_operations=%d",
        draft.job_id,
        len(draft.why_lines),
        len(scorable),
    )
    return draft
