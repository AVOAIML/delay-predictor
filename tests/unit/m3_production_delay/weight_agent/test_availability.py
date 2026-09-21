from m3_production_delay.llm_agents.weight_agent.availability import (
    ALL_SIGNALS_UNAVAILABLE_REASON,
    BOUNDS_WIDENED_REASON,
    apply_availability,
)
from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER, TOTAL_BP, Bounds

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


def test_all_available_is_a_no_op_redistribution():
    availability = {signal: True for signal in SIGNAL_ORDER}
    result = apply_availability(PRIOR_BP, BOUNDS_BP, availability, SIGNAL_ORDER, {})
    assert result.prior_bp == PRIOR_BP
    assert result.bounds_bp == BOUNDS_BP
    assert set(result.available_signals) == set(SIGNAL_ORDER)
    assert result.excluded_signals == ()


def test_excluded_signal_gets_exactly_zero():
    availability = {s: s != "seasonality" for s in SIGNAL_ORDER}
    result = apply_availability(
        PRIOR_BP, BOUNDS_BP, availability, SIGNAL_ORDER, {"seasonality": "insufficient_work_center_history"}
    )
    assert result.prior_bp["seasonality"] == 0
    assert result.bounds_bp["seasonality"] == Bounds(0, 0)
    assert len(result.excluded_signals) == 1
    assert result.excluded_signals[0].signal == "seasonality"
    assert result.excluded_signals[0].reason == "insufficient_work_center_history"


def test_redistribution_preserves_the_sum():
    availability = {s: s != "seasonality" for s in SIGNAL_ORDER}
    result = apply_availability(PRIOR_BP, BOUNDS_BP, availability, SIGNAL_ORDER, {})
    assert sum(result.prior_bp.values()) == TOTAL_BP


def test_bounds_scale_with_redistribution_and_remain_feasible():
    availability = {s: s != "seasonality" for s in SIGNAL_ORDER}
    result = apply_availability(PRIOR_BP, BOUNDS_BP, availability, SIGNAL_ORDER, {})
    available = result.available_signals
    sum_min = sum(result.bounds_bp[s].min for s in available)
    sum_max = sum(result.bounds_bp[s].max for s in available)
    assert sum_min <= TOTAL_BP <= sum_max
    # redistributed prior must itself sit within the redistributed bounds
    for signal in available:
        assert result.bounds_bp[signal].contains(result.prior_bp[signal])


def test_all_five_unavailable_is_handled_explicitly():
    availability = {s: False for s in SIGNAL_ORDER}
    result = apply_availability(PRIOR_BP, BOUNDS_BP, availability, SIGNAL_ORDER, {})
    assert result.available_signals == ()
    assert all(result.prior_bp[s] == 0 for s in SIGNAL_ORDER)
    assert all(result.bounds_bp[s] == Bounds(0, 0) for s in SIGNAL_ORDER)
    assert len(result.excluded_signals) == len(SIGNAL_ORDER)
    assert ALL_SIGNALS_UNAVAILABLE_REASON in result.reasons


def test_infeasible_scaled_bounds_are_widened_to_full_range():
    # supplier_reliability alone: its bound (300,1500) scaled to a total of
    # 10000 for a single signal becomes (10000*300/500, 10000*1500/500) =
    # (6000, 30000) clipped to 10000 -> (6000, 10000), which DOES bracket
    # 10000, so pick a case that provably can't: min-only bound tighter than
    # what a single remaining signal can reach is impossible to construct
    # from a min<=prior<=max feasible bound under this scaling rule, so
    # exercise the widening path directly through a signal whose own max is
    # below what redistribution demands.
    tight_bounds = dict(BOUNDS_BP)
    tight_bounds["supplier_reliability"] = Bounds(300, 400)  # cannot reach the redistributed 10000
    availability = {s: s == "supplier_reliability" for s in SIGNAL_ORDER}
    result = apply_availability(PRIOR_BP, tight_bounds, availability, SIGNAL_ORDER, {})
    assert result.available_signals == ("supplier_reliability",)
    assert result.bounds_bp["supplier_reliability"] == Bounds(0, TOTAL_BP)
    assert BOUNDS_WIDENED_REASON in result.reasons


def test_zero_prior_among_available_signals_splits_evenly_without_divide_by_zero():
    zero_prior = {s: 0 for s in SIGNAL_ORDER}
    availability = {s: s in ("time_overrun", "operator_skill") for s in SIGNAL_ORDER}
    result = apply_availability(zero_prior, BOUNDS_BP, availability, SIGNAL_ORDER, {})
    assert sum(result.prior_bp.values()) == TOTAL_BP
    assert result.prior_bp["time_overrun"] == TOTAL_BP // 2
    assert result.prior_bp["operator_skill"] == TOTAL_BP // 2
