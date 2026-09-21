"""§4.3 historical blend, hardened per Improvement 1: a single continuous
shrinkage curve that is ALSO a hard floor, at the same time, with no
contradiction between the two.

    effective_n = max(0, n - n_floor)
    lambda_bp   = 0                                       if effective_n == 0
                  10000 * effective_n / (effective_n + k)  otherwise

At or below the floor, effective_n is exactly 0 so lambda_bp is exactly 0 —
zero influence, by construction, not by a branch that could drift out of
sync with the formula. Immediately above the floor, effective_n is a small
positive integer and lambda_bp is a small positive number that grows
smoothly with n — no jump at the boundary, because the boundary is where the
*input* to the smooth curve crosses zero, not where the *formula* changes.
This is what stops a fitted provider trained on a handful of rows from
buying any influence over the blend at all, while still guaranteeing the
499-vs-501-work-orders cliff the un-floored version could produce near a
sufficiency threshold cannot happen once past the floor.
"""

from __future__ import annotations

from collections.abc import Sequence

from m3_production_delay.llm_agents.weight_agent.models import TOTAL_BP
from m3_production_delay.llm_agents.weight_agent.rounding import assign_remainder_canonical


def compute_lambda_bp(n: int, k: int, n_floor: int) -> int:
    """``n_floor`` is the usable-history admissibility floor
    (history_policy.compute_usable_floor) — a statistical "is there enough
    data at all" question, deliberately independent of ``k`` (which answers
    "how fast should trust grow once admissible"). Never derive one from
    the other.

    Rounds to the nearest integer bp (not floor) — ``round(10000 *
    effective_n / (effective_n + k))`` conceptually, computed with integer
    arithmetic throughout (``(numerator + denominator // 2) // denominator``)
    so no float ever enters the calculation, matching every other numeric
    function in this package.
    """
    effective_n = max(0, n - n_floor)
    if effective_n == 0:
        return 0
    denominator = effective_n + k
    if denominator <= 0:
        return 0
    numerator = TOTAL_BP * effective_n
    return (numerator + denominator // 2) // denominator


def blend_weights(
    prior_bp: dict[str, int],
    fitted_bp: dict[str, int],
    lambda_bp: int,
    order: Sequence[str],
) -> dict[str, int]:
    """``final_bp = (lambda_bp * fitted_bp + (10000 - lambda_bp) * prior_bp) / 10000``,
    rounded with the shared canonical-order remainder rule so the result sums
    to exactly 10000 regardless of per-signal floor-division truncation.

    The lambda_bp=0 / lambda_bp=10000 endpoints are short-circuited so they
    are byte-identical to the prior / fitted vector respectively, with no
    dependency on floor-division behaving exactly as expected at the edges.
    """
    if lambda_bp <= 0:
        return {signal: prior_bp[signal] for signal in order}
    if lambda_bp >= TOTAL_BP:
        return {signal: fitted_bp[signal] for signal in order}

    floors: dict[str, int] = {}
    for signal in order:
        numerator = lambda_bp * fitted_bp[signal] + (TOTAL_BP - lambda_bp) * prior_bp[signal]
        floors[signal] = numerator // TOTAL_BP
    residual = TOTAL_BP - sum(floors.values())
    return assign_remainder_canonical(floors, residual, order)
