"""§5 projection: turn a proposed adjustment into a bounds-respecting weight
vector that sums to exactly 10000bp, or fail explicitly.

Clipping and renormalising are not composable in one pass — renormalising can
push a clipped weight back outside its bound. This implements the specified
fixed point instead: clip once, then repeatedly distribute the residual
across signals proportional to their remaining headroom (capacity-aware, so a
signal already at its bound is never pushed past it), assigning any
floor-division leftover one basis point at a time in canonical order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from m3_production_delay.llm_agents.weight_agent.exceptions import (
    NonZeroSumAdjustmentError,
    ProjectionInvariantError,
)
from m3_production_delay.llm_agents.weight_agent.models import TOTAL_BP, Bounds
from m3_production_delay.llm_agents.weight_agent.rounding import ensure_finite_ints, assign_remainder_canonical


@dataclass(frozen=True)
class ProjectionOutcome:
    success: bool
    weights_bp: dict[str, int] | None
    reason: str | None
    iterations: int


def project_adjustment(
    prior_bp: dict[str, int],
    bounds_bp: dict[str, Bounds],
    adjustment_bp: dict[str, int],
    order: Sequence[str],
    *,
    max_iterations: int,
) -> ProjectionOutcome:
    ensure_finite_ints(prior_bp, "prior_bp")
    ensure_finite_ints(adjustment_bp, "adjustment_bp")
    adjustment_sum = sum(adjustment_bp.get(signal, 0) for signal in order)
    if adjustment_sum != 0:
        raise NonZeroSumAdjustmentError(
            f"adjustment_bp must sum to 0, got {adjustment_sum}"
        )

    candidate = {
        signal: bounds_bp[signal].clip(prior_bp[signal] + adjustment_bp.get(signal, 0))
        for signal in order
    }
    residual = TOTAL_BP - sum(candidate.values())

    iterations = 0
    while residual != 0 and iterations < max_iterations:
        sign = 1 if residual > 0 else -1
        if sign > 0:
            headroom = {signal: bounds_bp[signal].max - candidate[signal] for signal in order}
        else:
            headroom = {signal: candidate[signal] - bounds_bp[signal].min for signal in order}
        total_headroom = sum(headroom.values())
        if total_headroom == 0:
            break

        amount = min(abs(residual), total_headroom)
        shares = {signal: (amount * headroom[signal]) // total_headroom for signal in order}
        for signal in order:
            candidate[signal] += sign * shares[signal]

        leftover = amount - sum(shares.values())
        if leftover:
            remaining_capacity = {signal: headroom[signal] - shares[signal] for signal in order}
            bumped = assign_remainder_canonical(
                {signal: 0 for signal in order}, sign * leftover, order, capacity=remaining_capacity
            )
            for signal in order:
                candidate[signal] += bumped[signal]

        residual = TOTAL_BP - sum(candidate.values())
        iterations += 1

    if residual != 0:
        return ProjectionOutcome(
            success=False, weights_bp=None, reason="non_convergence", iterations=iterations
        )

    total = sum(candidate.values())
    if total != TOTAL_BP or any(not bounds_bp[signal].contains(candidate[signal]) for signal in order):
        raise ProjectionInvariantError(
            f"projection claimed convergence but sum={total} or a bound was "
            f"violated — candidate={candidate}, bounds={bounds_bp}"
        )

    return ProjectionOutcome(success=True, weights_bp=candidate, reason=None, iterations=iterations)
