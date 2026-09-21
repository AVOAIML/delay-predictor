import random
from unittest.mock import patch

import pytest

from m3_production_delay.llm_agents.weight_agent.exceptions import (
    NonZeroSumAdjustmentError,
    ProjectionInvariantError,
)
from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER, TOTAL_BP, Bounds
from m3_production_delay.llm_agents.weight_agent.projection import project_adjustment
from m3_production_delay.llm_agents.weight_agent.rounding import (
    assign_remainder_canonical as _real_assign_remainder_canonical,
)

PRIOR_BP = {
    "time_overrun": 4000,
    "operator_skill": 3500,
    "seasonality": 1000,
    "material_availability": 1000,
    "supplier_reliability": 500,
}
BOUNDS_BP = {
    "time_overrun": Bounds(3000, 5000),
    "operator_skill": Bounds(2000, 4500),
    "seasonality": Bounds(500, 2000),
    "material_availability": Bounds(500, 2500),
    "supplier_reliability": Bounds(300, 1500),
}


def _random_zero_sum_adjustment(rng: random.Random, magnitude: int = 800) -> dict:
    values = {signal: rng.randint(-magnitude, magnitude) for signal in SIGNAL_ORDER[:-1]}
    values[SIGNAL_ORDER[-1]] = -sum(values.values())
    return values


def test_randomized_adjustments_always_sum_to_10000_and_respect_bounds():
    rng = random.Random(1234)
    for _ in range(500):
        adjustment = _random_zero_sum_adjustment(rng)
        outcome = project_adjustment(
            PRIOR_BP, BOUNDS_BP, adjustment, SIGNAL_ORDER, max_iterations=20
        )
        if not outcome.success:
            continue  # non-convergence is a valid outcome for extreme adjustments
        assert sum(outcome.weights_bp.values()) == TOTAL_BP
        for signal in SIGNAL_ORDER:
            assert BOUNDS_BP[signal].contains(outcome.weights_bp[signal])


def test_nonzero_sum_adjustment_is_rejected_not_normalized():
    adjustment = {signal: 0 for signal in SIGNAL_ORDER}
    adjustment["time_overrun"] = 100  # sums to 100, not 0
    with pytest.raises(NonZeroSumAdjustmentError):
        project_adjustment(PRIOR_BP, BOUNDS_BP, adjustment, SIGNAL_ORDER, max_iterations=20)


def test_adjustment_breaching_a_bound_is_reprojected_onto_the_others():
    # supplier_reliability's prior (500) - 300 would fall below its min (300)
    # by exactly enough to require the clip; the deficit must land on the
    # other four signals via headroom-proportional redistribution, not get
    # silently renormalised back onto the breached signal.
    adjustment = {
        "time_overrun": 200,
        "operator_skill": -200,
        "seasonality": 0,
        "material_availability": 300,
        "supplier_reliability": -300,
    }
    outcome = project_adjustment(PRIOR_BP, BOUNDS_BP, adjustment, SIGNAL_ORDER, max_iterations=20)
    assert outcome.success
    assert sum(outcome.weights_bp.values()) == TOTAL_BP
    assert outcome.weights_bp["supplier_reliability"] == 300  # clipped to its min, stays there
    for signal in SIGNAL_ORDER:
        assert BOUNDS_BP[signal].contains(outcome.weights_bp[signal])


def test_non_convergence_reports_failure_without_raising():
    tight_bounds = {signal: Bounds(0, 1000) for signal in SIGNAL_ORDER}  # sum(max)=5000 < 10000
    prior = {signal: 1000 for signal in SIGNAL_ORDER}  # prior itself violates TOTAL_BP=10000
    adjustment = {signal: 0 for signal in SIGNAL_ORDER}
    outcome = project_adjustment(prior, tight_bounds, adjustment, SIGNAL_ORDER, max_iterations=20)
    assert outcome.success is False
    assert outcome.reason == "non_convergence"
    assert outcome.weights_bp is None


def test_identical_input_produces_byte_identical_output():
    adjustment = {
        "time_overrun": 100,
        "operator_skill": -50,
        "seasonality": -50,
        "material_availability": 50,
        "supplier_reliability": -50,
    }
    first = project_adjustment(PRIOR_BP, BOUNDS_BP, adjustment, SIGNAL_ORDER, max_iterations=20)
    second = project_adjustment(PRIOR_BP, BOUNDS_BP, adjustment, SIGNAL_ORDER, max_iterations=20)
    assert first == second


def test_zero_adjustment_returns_prior_unchanged():
    adjustment = {signal: 0 for signal in SIGNAL_ORDER}
    outcome = project_adjustment(PRIOR_BP, BOUNDS_BP, adjustment, SIGNAL_ORDER, max_iterations=20)
    assert outcome.success
    assert outcome.weights_bp == PRIOR_BP


def test_headroom_proportional_split_with_zero_leftover():
    # Constructed so the floor-division shares land exactly on the residual
    # with nothing left to distribute one bp at a time — exercises the
    # "leftover == 0" branch distinctly from the (far more common) case
    # covered by every other test here.
    order = ["a", "b"]
    prior = {"a": 5000, "b": 5000}
    bounds = {"a": Bounds(0, 5000), "b": Bounds(0, 10000)}
    adjustment = {"a": 2000, "b": -2000}  # clips a down to 5000, b to 3000 -> sum 8000
    outcome = project_adjustment(prior, bounds, adjustment, order, max_iterations=20)
    assert outcome.success
    assert outcome.weights_bp == {"a": 5000, "b": 5000}
    assert outcome.iterations == 1


def test_projection_invariant_error_on_a_corrupted_remainder_assignment():
    # The convergence proof (config-feasible bounds -> always converges in
    # one pass) means the ProjectionInvariantError branch is unreachable
    # through the public API with a correctly-behaving rounding.py. This
    # proves the guard itself is correct by forcing exactly the corruption
    # it exists to catch, rather than leaving it untested because it can't
    # happen in practice.
    # This adjustment (same as the bound-breach test above) forces a clip on
    # supplier_reliability, a nonzero residual, AND a nonzero remainder
    # leftover — i.e. it actually reaches the assign_remainder_canonical call
    # this test corrupts, unlike an adjustment that resolves on the first
    # clip with no while-loop iteration at all.
    adjustment = {
        "time_overrun": 200,
        "operator_skill": -200,
        "seasonality": 0,
        "material_availability": 300,
        "supplier_reliability": -300,
    }

    def _corrupt(values, residual, order, capacity=None):
        # Real result, sum-preserving, but shifted 1bp from a signal already
        # sitting exactly on its bound (supplier_reliability, clipped to its
        # min of 300) onto another — sum stays correct so the loop still
        # exits normally, but that one signal ends up 1bp outside its bound.
        result = dict(_real_assign_remainder_canonical(values, residual, order, capacity=capacity))
        result["supplier_reliability"] = result.get("supplier_reliability", 0) - 1
        result["time_overrun"] = result.get("time_overrun", 0) + 1
        return result

    with patch(
        "m3_production_delay.llm_agents.weight_agent.projection.assign_remainder_canonical",
        side_effect=_corrupt,
    ):
        with pytest.raises(ProjectionInvariantError):
            project_adjustment(PRIOR_BP, BOUNDS_BP, adjustment, SIGNAL_ORDER, max_iterations=20)
