"""§4.2 signal availability mask. Applied to every non-configured resolution
path, before any adjustment or blend — an unavailable signal must never
silently reduce the maximum achievable risk score, so its share of the prior
is redistributed to the signals that remain, and its bounds are scaled by the
same factor rather than left to skew the feasible region.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from m3_production_delay.llm_agents.weight_agent.models import TOTAL_BP, Bounds, ExcludedSignal
from m3_production_delay.llm_agents.weight_agent.rounding import assign_remainder_canonical, ensure_finite_ints

ALL_SIGNALS_UNAVAILABLE_REASON = "all_signals_unavailable"
BOUNDS_WIDENED_REASON = "redistributed_bounds_infeasible_widened_to_full_range"


@dataclass(frozen=True)
class AvailabilityResult:
    prior_bp: dict[str, int]  # the redistributed prior — the baseline every later stage uses
    bounds_bp: dict[str, Bounds]  # scaled by the same redistribution factor
    available_signals: tuple[str, ...]
    excluded_signals: tuple[ExcludedSignal, ...]
    reasons: tuple[str, ...]  # e.g. bounds-widened note; empty in the common case


def apply_availability(
    prior_bp: dict[str, int],
    bounds_bp: dict[str, Bounds],
    availability: dict[str, bool],
    order: Sequence[str],
    exclusion_reasons: dict[str, str],
) -> AvailabilityResult:
    """``availability[signal]`` is an explicit bool supplied by the caller —
    this function never infers availability itself."""
    ensure_finite_ints(prior_bp, "prior_bp")
    available = [signal for signal in order if availability.get(signal, True)]
    excluded = [signal for signal in order if signal not in available]
    excluded_signals = tuple(
        ExcludedSignal(signal, exclusion_reasons.get(signal, "unavailable")) for signal in excluded
    )

    if not available:
        return AvailabilityResult(
            prior_bp={signal: 0 for signal in order},
            bounds_bp={signal: Bounds(0, 0) for signal in order},
            available_signals=(),
            excluded_signals=tuple(
                ExcludedSignal(signal, exclusion_reasons.get(signal, ALL_SIGNALS_UNAVAILABLE_REASON))
                for signal in order
            ),
            reasons=(ALL_SIGNALS_UNAVAILABLE_REASON,),
        )

    available_order = [signal for signal in order if signal in available]
    total_prior = sum(prior_bp[signal] for signal in available_order)

    if total_prior > 0:
        floors = {signal: (prior_bp[signal] * TOTAL_BP) // total_prior for signal in available_order}
    else:
        # Degenerate: every available signal has a zero prior. Split evenly
        # rather than dividing by zero — still deterministic via canonical
        # order for the remainder.
        floors = {signal: TOTAL_BP // len(available_order) for signal in available_order}
    residual = TOTAL_BP - sum(floors.values())
    redistributed_available = assign_remainder_canonical(floors, residual, available_order)
    redistributed_prior = {signal: 0 for signal in excluded}
    redistributed_prior.update(redistributed_available)

    scale_den = total_prior if total_prior > 0 else TOTAL_BP
    scaled_bounds: dict[str, Bounds] = {signal: Bounds(0, 0) for signal in excluded}
    for signal in available_order:
        bound = bounds_bp[signal]
        lo = (bound.min * TOTAL_BP) // scale_den
        hi = (bound.max * TOTAL_BP) // scale_den
        scaled_bounds[signal] = Bounds(min(lo, TOTAL_BP), min(hi, TOTAL_BP))

    sum_min = sum(scaled_bounds[signal].min for signal in available_order)
    sum_max = sum(scaled_bounds[signal].max for signal in available_order)
    reasons: tuple[str, ...] = ()
    if not sum_min <= TOTAL_BP <= sum_max:
        for signal in available_order:
            scaled_bounds[signal] = Bounds(0, TOTAL_BP)
        reasons = (BOUNDS_WIDENED_REASON,)

    return AvailabilityResult(
        prior_bp=redistributed_prior,
        bounds_bp=scaled_bounds,
        available_signals=tuple(available_order),
        excluded_signals=excluded_signals,
        reasons=reasons,
    )
