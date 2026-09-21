"""Deterministic checks that run BEFORE the judge, over the composed draft and
the evidence it was composed from.

These are not a second opinion on the LLM — they run first, and an ``error``
from any of them means the draft is a composer or evidence bug, not a wording
problem an LLM could be asked about. The pipeline rejects such a draft outright
rather than spending a call on it or retrying it.

The division of labour with the judge is deliberate and total:

  * **Here (deterministic):** does the signal exist and fire, is anything
    fired and weighted missing, is every numeral in the prose re-derivable
    from the evidence, is the summary consistent with the score and the
    tenant's threshold, is a zero-weight signal being shown as a cause, does a
    line name a cause this system cannot observe, and is the underlying data
    physically plausible.
  * **The judge (LLM):** given evidence that has already passed all of the
    above, is each line's *claim* actually supported by it.

Only ``error`` rejects. ``warning`` downgrades an approval to
``approved_with_warnings`` and is carried into the payload; ``info`` is
context for the reader and changes nothing.
"""

from __future__ import annotations

import re

from maxxflow_core.errors import get_logger

from m3_production_delay.review.composer import MAX_NAMED_COMPONENTS
from m3_production_delay.review.evidence import FIRE_BASELINES
from m3_production_delay.review.schemas import (
    SCOPE_JOB,
    SCOPE_OPERATION,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_PREDECESSOR_OVERRUN,
    SIGNAL_SUPPLIER_RELIABILITY,
    SIGNAL_TIME_OVERRUN,
    EvidencePack,
    InsightDraft,
    InsightLine,
    Issue,
    OperationEvidence,
    SignalEvidence,
)

log = get_logger("m3_production_delay.review.validators")

#: Causes this platform cannot observe. None of them is derivable from any
#: signal the rule engine computes, so a line naming one is by definition
#: unsupported — whether a template drifted or a future caller hand-wrote a
#: line, the answer is the same.
FORBIDDEN_CAUSE_PATTERN = re.compile(
    r"breakdown|quality hold|rework|approval|inspection|weather|strike", re.IGNORECASE
)

#: Numerals as rendered by the composer's formatters, including the delta's
#: leading sign.
_NUMERAL_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")

#: Relative tolerance on a quoted number, with an absolute floor equal to half
#: the coarsest display step the composer uses (one decimal place on hours).
#: Without the floor, correct rounding of a small number — 0.06 hrs rendered
#: as "0.1" — would read as a 67% misquote.
QUOTE_RELATIVE_TOLERANCE = 0.02
QUOTE_ABSOLUTE_FLOOR = 0.05

#: Statuses that mean no work has started. An operation in one of these that
#: the engine nevertheless scored as delayed is worth flagging: the composite
#: got there on operator, material and supplier history alone.
NOT_STARTED_STATUSES: frozenset[str] = frozenset({"NOT_STARTED", "TO_DO"})

#: Ratio of logged to planned time beyond which the two are more likely to be
#: in different units than genuinely that far apart. 20x is far past anything
#: a real overrun produces and comfortably below the 60x a minutes/seconds
#: mix-up would show.
UNIT_MISMATCH_RATIO = 20.0


def _within_tolerance(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(abs(expected) * QUOTE_RELATIVE_TOLERANCE,
                                         QUOTE_ABSOLUTE_FLOOR)


def _signal_for(pack: EvidencePack, line: InsightLine) -> SignalEvidence | None:
    if line.scope == SCOPE_JOB:
        return pack.job_signal(line.signal_key)
    op = pack.operation(line.operation_id or "")
    return op.signal(line.signal_key) if op is not None else None


def _entity_fragments(pack: EvidencePack) -> list[str]:
    """Text that legitimately contains digits without being a measurement —
    job references like ``WH/MO/00142``, part names like ``Steel Plate 12mm``,
    operation labels ending in a short id. Stripped before numerals are
    extracted so a part number can never be read as a misquoted figure."""
    fragments: list[str] = [pack.job_id]
    for op in pack.operations:
        fragments.append(op.operation_name)
        for component in op.components_short:
            fragments.extend(
                [component.component_id, component.name or "", component.vendor_name or ""]
            )
    for component in pack.material_overrun:
        fragments.extend(
            [component.component_id, component.name or "", component.vendor_name or ""]
        )
    for signal in pack.job_signals:
        fragments.extend(str(name) for name in (signal.detail.get("vendor_names") or []))
    # Longest first: stripping "Steel Plate 12" before "Steel Plate" would
    # leave a stray "12" behind.
    return sorted({f for f in fragments if f}, key=len, reverse=True)


def _numerals(text: str, fragments: list[str]) -> list[float]:
    stripped = text
    for fragment in fragments:
        stripped = stripped.replace(fragment, " ")
    return [float(match) for match in _NUMERAL_PATTERN.findall(stripped)]


def _expected_quotes(pack: EvidencePack, line: InsightLine) -> dict[str, float] | None:
    """Re-derive, from the evidence alone, the numbers this line is allowed to
    quote. Returns None when the line's signal has no evidence at all — that
    is ``check_signals_fired``'s finding to report, not this one's."""
    signal = _signal_for(pack, line)
    if signal is None:
        return None

    if line.signal_key == SIGNAL_TIME_OVERRUN:
        op = pack.operation(line.operation_id or "")
        if op is None or op.actual_duration_minutes is None or op.expected_duration_minutes is None:
            return {}
        actual_hrs = op.actual_duration_minutes / 60.0
        expected_hrs = op.expected_duration_minutes / 60.0
        return {
            "actual_hrs": actual_hrs,
            "expected_hrs": expected_hrs,
            "delta_hrs": actual_hrs - expected_hrs,
        }

    if line.signal_key == SIGNAL_OPERATOR_PACE:
        return {} if signal.value is None else {"pace_ratio": signal.value}

    if line.signal_key == SIGNAL_MATERIAL_SHORTFALL:
        quotable: dict[str, float] = {}
        named = [c for c in pack.material_overrun if c.shortfall_quantity is not None]
        for component in named[:MAX_NAMED_COMPONENTS]:
            quotable[f"shortfall:{component.component_id}"] = component.shortfall_quantity
        remaining = len(pack.material_overrun) - len(named[:MAX_NAMED_COMPONENTS])
        if remaining > 0:
            quotable["more_count"] = float(remaining)
        return quotable

    if line.signal_key == SIGNAL_SUPPLIER_RELIABILITY:
        return {} if signal.value is None else {"mean_lead_time_ratio": signal.value}

    return {}


def _renderable(pack: EvidencePack, op: OperationEvidence | None, key: str) -> bool:
    """Whether the composer could have written a line for this signal without
    inventing a number. Mirrors the composer's own guards — a signal that
    fired but whose supporting fields are missing is a warning, not the
    composer omitting something it could have said."""
    if key == SIGNAL_TIME_OVERRUN:
        if op is None:
            return False
        signal = op.signal(key)
        detail = signal.detail if signal else {}
        return all(detail.get(name) is not None for name in ("actual_hrs", "expected_hrs", "delta_hrs"))
    if key == SIGNAL_OPERATOR_PACE:
        return op is not None and op.signal(key) is not None and op.signal(key).value is not None
    if key == SIGNAL_MATERIAL_SHORTFALL:
        return any(c.shortfall_quantity is not None for c in pack.material_overrun)
    if key == SIGNAL_SUPPLIER_RELIABILITY:
        signal = pack.job_signal(key)
        return bool(
            signal is not None
            and signal.value is not None
            and (signal.detail.get("vendor_names") or [])
        )
    return False


# --- individual checks ------------------------------------------------------


def check_signals_fired(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    """Every line must name a signal that exists in the evidence for its scope
    and that actually cleared its fire baseline."""
    issues: list[Issue] = []
    for line in draft.why_lines:
        signal = _signal_for(pack, line)
        if signal is None:
            issues.append(
                Issue(
                    check="signal_missing",
                    severity=SEVERITY_ERROR,
                    message=(
                        f"line {line.index} cites {line.signal_key!r}, which has no evidence at "
                        f"{line.scope} scope"
                    ),
                    ref=str(line.index),
                )
            )
            continue
        if not signal.fired:
            issues.append(
                Issue(
                    check="signal_not_fired",
                    severity=SEVERITY_ERROR,
                    message=(
                        f"line {line.index} cites {line.signal_key!r} (value={signal.value!r}), "
                        f"which did not clear its baseline of "
                        f"{FIRE_BASELINES[line.signal_key]}"
                    ),
                    ref=str(line.index),
                )
            )
    return issues


def check_no_omission(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    """Nothing that fired, carried weight, and could be stated may be left
    out — an explanation that silently drops the second-biggest cause is
    misleading even when every line in it is true."""
    issues: list[Issue] = []
    shown = {(line.signal_key, line.operation_id) for line in draft.why_lines}

    def record(key: str, operation_id: str | None, op: OperationEvidence | None) -> None:
        if (key, operation_id) in shown:
            return
        if _renderable(pack, op, key):
            issues.append(
                Issue(
                    check="omitted_signal",
                    severity=SEVERITY_ERROR,
                    message=f"{key!r} fired with weight but no line was composed for it",
                    ref=operation_id,
                )
            )
        else:
            issues.append(
                Issue(
                    check="omitted_signal_unrenderable",
                    severity=SEVERITY_WARNING,
                    message=(
                        f"{key!r} fired with weight but the evidence needed to state it is "
                        "incomplete, so no line was composed"
                    ),
                    ref=operation_id,
                )
            )

    for op in pack.scorable_operations:
        for key in (SIGNAL_TIME_OVERRUN, SIGNAL_OPERATOR_PACE):
            signal = op.signal(key)
            if signal is not None and signal.fired and signal.is_weighted:
                record(key, op.operation_id, op)

    if pack.scorable_operations:
        for key in (SIGNAL_MATERIAL_SHORTFALL, SIGNAL_SUPPLIER_RELIABILITY):
            signal = pack.job_signal(key)
            if signal is not None and signal.fired and signal.is_weighted:
                record(key, None, None)
    return issues


def check_quoted_numbers(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    """Two directions, both required: every number a line claims to quote must
    match the evidence, and every numeral rendered into the prose must be one
    of those quoted values. Together they make the text re-derivable rather
    than merely plausible."""
    issues: list[Issue] = []
    fragments = _entity_fragments(pack)
    for line in draft.why_lines:
        expected = _expected_quotes(pack, line)
        if expected is None:
            continue  # no evidence at all — check_signals_fired owns this
        for name, value in line.quoted.items():
            if name not in expected:
                issues.append(
                    Issue(
                        check="quoted_number_unknown",
                        severity=SEVERITY_ERROR,
                        message=(
                            f"line {line.index} quotes {name!r}={value}, which is not a field of "
                            f"{line.signal_key!r}'s evidence"
                        ),
                        ref=str(line.index),
                    )
                )
            elif not _within_tolerance(value, expected[name]):
                issues.append(
                    Issue(
                        check="quoted_number_mismatch",
                        severity=SEVERITY_ERROR,
                        message=(
                            f"line {line.index} quotes {name!r}={value}, but the evidence says "
                            f"{expected[name]}"
                        ),
                        ref=str(line.index),
                    )
                )
        allowed = list(line.quoted.values())
        text = f"{line.headline} {line.detail} {line.delta or ''}"
        for numeral in _numerals(text, fragments):
            if not any(_within_tolerance(numeral, value) for value in allowed):
                issues.append(
                    Issue(
                        check="unquoted_number",
                        severity=SEVERITY_ERROR,
                        message=(
                            f"line {line.index} renders {numeral}, which matches no value it "
                            "quotes from the evidence"
                        ),
                        ref=str(line.index),
                    )
                )
    return issues


def check_summary_consistent(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    """The badge must agree with the evidence and with the tenant's
    threshold — the one number a reader acts on before reading any line."""
    issues: list[Issue] = []
    pairs = (
        ("summary_risk_score", draft.summary_risk_score, pack.summary_risk_score),
        ("summary_overrun_hours", draft.summary_overrun_hours, pack.summary_overrun_hours),
    )
    for name, drafted, evidenced in pairs:
        if (drafted is None) != (evidenced is None) or (
            drafted is not None and evidenced is not None and not _within_tolerance(drafted, evidenced)
        ):
            issues.append(
                Issue(
                    check="summary_mismatch",
                    severity=SEVERITY_ERROR,
                    message=f"{name}={drafted!r} does not match the evidence ({evidenced!r})",
                )
            )
    if draft.summary_is_delayed != pack.summary_is_delayed:
        issues.append(
            Issue(
                check="summary_mismatch",
                severity=SEVERITY_ERROR,
                message=(
                    f"summary_is_delayed={draft.summary_is_delayed!r} does not match the evidence "
                    f"({pack.summary_is_delayed!r})"
                ),
            )
        )
    if draft.summary_risk_score is not None and draft.summary_is_delayed is not None:
        expected = draft.summary_risk_score > pack.delay_threshold
        if draft.summary_is_delayed != expected:
            issues.append(
                Issue(
                    check="threshold_inconsistent",
                    severity=SEVERITY_ERROR,
                    message=(
                        f"summary_is_delayed={draft.summary_is_delayed!r} contradicts "
                        f"score {draft.summary_risk_score} against threshold "
                        f"{pack.delay_threshold}"
                    ),
                )
            )
    return issues


def check_zero_weight_hidden(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    """A signal the tenant weights at zero did not move the score, so it may
    not be shown as a cause. This is what keeps a cascading predecessor —
    computed by the engine, never weighted — out of the why-lines."""
    issues: list[Issue] = []
    for line in draft.why_lines:
        signal = _signal_for(pack, line)
        if signal is not None and not signal.is_weighted:
            issues.append(
                Issue(
                    check="zero_weight_signal_shown",
                    severity=SEVERITY_ERROR,
                    message=(
                        f"line {line.index} presents {line.signal_key!r} as a cause, but its "
                        f"weight is {signal.weight}"
                    ),
                    ref=str(line.index),
                )
            )
    return issues


def check_forbidden_causes(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    issues: list[Issue] = []
    for line in draft.why_lines:
        text = f"{line.headline} {line.detail} {line.delta or ''}"
        match = FORBIDDEN_CAUSE_PATTERN.search(text)
        if match:
            issues.append(
                Issue(
                    check="forbidden_cause",
                    severity=SEVERITY_ERROR,
                    message=(
                        f"line {line.index} names {match.group(0)!r} as a cause; no signal in this "
                        "system can observe it"
                    ),
                    ref=str(line.index),
                )
            )
    return issues


def check_plausibility(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    """Warnings about the underlying data rather than the prose. Neither of
    these makes the insight wrong — both make it worth a second look."""
    issues: list[Issue] = []
    for op in pack.operations:
        actual = op.actual_duration_minutes
        expected = op.expected_duration_minutes
        if actual is not None and expected and actual > UNIT_MISMATCH_RATIO * expected:
            issues.append(
                Issue(
                    check="implausible_duration",
                    severity=SEVERITY_WARNING,
                    message=(
                        f"{op.operation_name} logged {actual / 60.0:.1f} hrs against "
                        f"{expected / 60.0:.1f} planned (>{UNIT_MISMATCH_RATIO:.0f}x) — possible "
                        "unit mismatch"
                    ),
                    ref=op.operation_id,
                )
            )
        if (op.status or "").upper() in NOT_STARTED_STATUSES and op.is_delayed:
            issues.append(
                Issue(
                    check="not_started_but_delayed",
                    severity=SEVERITY_WARNING,
                    message=(
                        f"{op.operation_name} is {op.status} yet scored as delayed "
                        f"({op.composite_risk_score!r} against threshold {pack.delay_threshold})"
                    ),
                    ref=op.operation_id,
                )
            )
    return issues


def check_zero_fired(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    """The job is flagged delayed but nothing cleared a fire baseline.

    Reachable and not a bug: the composite is a weighted mean of raw ratios,
    so a single signal sitting between 1.0 and its own baseline (an operator
    pace of 1.15, say) can carry the mean past the threshold on its own. The
    badge stands — it is the engine's, not this agent's — but there is no
    cause to state, so the warning names the strongest signal instead.
    """
    if not draft.summary_is_delayed or draft.why_lines or not pack.scorable_operations:
        return []
    ranked = [
        (signal, op.operation_id)
        for op in pack.scorable_operations
        for signal in op.signals
        if signal.value is not None and signal.is_weighted
    ]
    ranked += [(signal, None) for signal in pack.job_signals if signal.value is not None]
    if not ranked:
        return [
            Issue(
                check="no_fired_signal",
                severity=SEVERITY_WARNING,
                message="job is flagged delayed but no signal has a value to explain it",
            )
        ]
    top, ref = max(
        ranked,
        key=lambda pair: (
            pair[0].contribution if pair[0].contribution is not None else -1.0,
            pair[0].value or 0.0,
        ),
    )
    return [
        Issue(
            check="no_fired_signal",
            severity=SEVERITY_WARNING,
            message=(
                f"job is flagged delayed but no signal cleared its baseline; the strongest is "
                f"{top.key!r} at {top.value} (baseline {FIRE_BASELINES[top.key]})"
            ),
            ref=ref,
        )
    ]


def check_cascade_context(pack: EvidencePack, draft: InsightDraft) -> list[Issue]:
    """A predecessor running long is context, never a cause: it is not a
    weighted input to the score, so it cannot appear as a why-line. Carried as
    ``info`` so the panel can still say the delay is inherited."""
    issues: list[Issue] = []
    for op in pack.operations:
        signal = op.signal(SIGNAL_PREDECESSOR_OVERRUN)
        if signal is not None and signal.fired:
            issues.append(
                Issue(
                    check="cascading_predecessor",
                    severity=SEVERITY_INFO,
                    message=(
                        f"{op.operation_name} depends on an operation running at "
                        f"{signal.value:.2f}× its planned duration"
                    ),
                    ref=op.operation_id,
                )
            )
    return issues


#: Run in this order; the output order is what the payload carries.
CHECKS = (
    check_signals_fired,
    check_no_omission,
    check_quoted_numbers,
    check_summary_consistent,
    check_zero_weight_hidden,
    check_forbidden_causes,
    check_plausibility,
    check_zero_fired,
    check_cascade_context,
)


def validate(pack: EvidencePack, draft: InsightDraft) -> tuple[Issue, ...]:
    """Every deterministic check, in order. Any ``error`` means the draft must
    not be published as an explanation — see ``pipeline.review_job``."""
    issues: list[Issue] = []
    for check in CHECKS:
        issues.extend(check(pack, draft))
    errors = sum(1 for issue in issues if issue.is_error)
    if errors:
        log.warning(
            "m3_review validation failed job_id=%s errors=%d issues=%d",
            pack.job_id,
            errors,
            len(issues),
        )
    return tuple(issues)
