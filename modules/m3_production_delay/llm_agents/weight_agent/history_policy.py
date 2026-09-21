"""§6 history admissibility policy, extended per Improvement 1 into four
explicit tiers instead of one boolean:

    inadmissible  -- at or below the usable floor (compute_usable_floor).
                     Fitted weights are forbidden outright; blend.compute_lambda_bp
                     returns exactly 0 for these inputs, and the resolver never
                     even attempts a blend.
    usable_weak   -- above the floor but below the admin-facing sufficiency
                     bundle (min_history_span_days / min_completed_work_orders /
                     min_delayed_work_orders / min_signal_coverage). Gets a
                     small, continuously-scaled fitted contribution.
    sufficient    -- clears the full bundle.
    preferred     -- clears the bundle AND the (higher) preferred history span.

Rows existing is not sufficiency: every criterion in the bundle is checked
independently and every FAILING one is recorded, even once the overall tier
is already known.

The usable floor answers "is there enough data to estimate these weights at
all" — a statistical admissibility question, sized by the number of free
weight parameters (see :func:`compute_usable_floor`). This is a distinct
question from ``shrinkage_k`` (blend.compute_lambda_bp): once admissible,
how fast trust in the fitted vector should grow with more data. The two must
never be derived from one another.
"""

from __future__ import annotations

from m3_production_delay.llm_agents.weight_agent.config import HistoryPolicyConfig
from m3_production_delay.llm_agents.weight_agent.models import (
    ADMISSIBILITY_INADMISSIBLE,
    ADMISSIBILITY_PREFERRED,
    ADMISSIBILITY_SUFFICIENT,
    ADMISSIBILITY_USABLE_WEAK,
    SIGNAL_ORDER,
    HistoryAssessment,
    HistoryInputs,
)

HISTORY_SPAN_BELOW_MINIMUM = "history_span_below_minimum"
COMPLETED_WORK_ORDERS_BELOW_MINIMUM = "completed_work_orders_below_minimum"
DELAYED_WORK_ORDERS_BELOW_MINIMUM = "delayed_work_orders_below_minimum"
SIGNAL_COVERAGE_BELOW_MINIMUM = "signal_coverage_below_minimum"
BELOW_USABLE_FLOOR = "delayed_work_orders_at_or_below_usable_floor"


def compute_usable_floor(available_count: int, events_per_parameter: int) -> int:
    """Minimum delayed work orders before a fitted estimate is admissible.

    The weight-sum-to-10000 constraint removes one degree of freedom, so
    ``available_count`` signals have ``available_count - 1`` free weight
    parameters to identify — not always 5-1=4: a tenant missing signals has
    fewer parameters to estimate and so needs less data, and a tenant with
    exactly one available signal has ZERO free parameters (the only valid
    weight is 100% on that signal — there is nothing to optimize between
    signals), so the floor is correctly 0 with no special-casing needed.
    """
    free_parameters = max(0, available_count - 1)
    return events_per_parameter * free_parameters


def assess_history(
    inputs: HistoryInputs, policy: HistoryPolicyConfig, available_count: int
) -> HistoryAssessment:
    reasons: list[str] = []

    if inputs.history_span_days < policy.min_history_span_days:
        reasons.append(HISTORY_SPAN_BELOW_MINIMUM)
    if inputs.completed_work_orders < policy.min_completed_work_orders:
        reasons.append(COMPLETED_WORK_ORDERS_BELOW_MINIMUM)
    if inputs.delayed_work_orders < policy.min_delayed_work_orders:
        reasons.append(DELAYED_WORK_ORDERS_BELOW_MINIMUM)

    low_coverage_signals = [
        signal
        for signal in SIGNAL_ORDER
        if inputs.per_signal_coverage.get(signal, 0.0) < policy.min_signal_coverage
    ]
    if low_coverage_signals:
        reasons.append(SIGNAL_COVERAGE_BELOW_MINIMUM)

    usable_floor = compute_usable_floor(available_count, policy.events_per_parameter)
    if inputs.delayed_work_orders <= usable_floor:
        admissibility = ADMISSIBILITY_INADMISSIBLE
        if BELOW_USABLE_FLOOR not in reasons:
            reasons = [BELOW_USABLE_FLOOR, *reasons]
    elif reasons:
        admissibility = ADMISSIBILITY_USABLE_WEAK
    elif inputs.history_span_days >= policy.preferred_history_span_days:
        admissibility = ADMISSIBILITY_PREFERRED
    else:
        admissibility = ADMISSIBILITY_SUFFICIENT

    return HistoryAssessment(admissibility=admissibility, reasons=tuple(reasons), lambda_bp=0)
