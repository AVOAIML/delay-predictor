from m3_production_delay.llm_agents.weight_agent.config import HistoryPolicyConfig
from m3_production_delay.llm_agents.weight_agent.history_policy import (
    BELOW_USABLE_FLOOR,
    COMPLETED_WORK_ORDERS_BELOW_MINIMUM,
    DELAYED_WORK_ORDERS_BELOW_MINIMUM,
    HISTORY_SPAN_BELOW_MINIMUM,
    SIGNAL_COVERAGE_BELOW_MINIMUM,
    assess_history,
    compute_usable_floor,
)
from m3_production_delay.llm_agents.weight_agent.models import (
    ADMISSIBILITY_INADMISSIBLE,
    ADMISSIBILITY_PREFERRED,
    ADMISSIBILITY_SUFFICIENT,
    ADMISSIBILITY_USABLE_WEAK,
    SIGNAL_ORDER,
    HistoryInputs,
)

POLICY = HistoryPolicyConfig()  # events_per_parameter=10, min_delayed_work_orders=50


def _inputs(**overrides) -> HistoryInputs:
    defaults = dict(
        history_span_days=400,
        completed_work_orders=600,
        delayed_work_orders=80,
        per_signal_coverage={signal: 0.9 for signal in SIGNAL_ORDER},
    )
    defaults.update(overrides)
    return HistoryInputs(**defaults)


# --- compute_usable_floor: signal-count-aware, not a fixed constant --------


def test_five_available_signals_gives_four_free_parameters():
    assert compute_usable_floor(5, events_per_parameter=10) == 40


def test_four_available_signals_gives_three_free_parameters():
    assert compute_usable_floor(4, events_per_parameter=10) == 30


def test_three_available_signals_gives_two_free_parameters():
    assert compute_usable_floor(3, events_per_parameter=10) == 20


def test_two_available_signals_gives_one_free_parameter():
    assert compute_usable_floor(2, events_per_parameter=10) == 10


def test_one_available_signal_has_zero_free_parameters_and_zero_floor():
    # Only one valid weight exists (100% on that signal) — there is nothing
    # to optimize between signals, so no statistical floor applies. This
    # falls out of the formula itself, no special-casing needed.
    assert compute_usable_floor(1, events_per_parameter=10) == 0


def test_zero_available_signals_does_not_crash_even_though_unreachable_in_practice():
    # The resolver raises AllSignalsUnavailableError before this could ever
    # be called with 0 in production — this just proves the pure function
    # itself doesn't misbehave (negative floor, exception, etc).
    assert compute_usable_floor(0, events_per_parameter=10) == 0


def test_floor_scales_with_events_per_parameter():
    assert compute_usable_floor(5, events_per_parameter=1) == 4
    assert compute_usable_floor(5, events_per_parameter=20) == 80


# --- assess_history: floor now depends on available_count ------------------


def test_all_criteria_met_but_below_preferred_span_is_sufficient_with_no_reasons():
    # 200 days clears min_history_span_days (180) but not the higher
    # preferred_history_span_days (365) — the "sufficient" tier proper.
    # delayed_work_orders=80 is above the 5-signal floor (40) and the
    # sufficiency bar (50).
    result = assess_history(_inputs(history_span_days=200), POLICY, available_count=5)
    assert result.admissibility == ADMISSIBILITY_SUFFICIENT
    assert result.sufficient is True
    assert result.reasons == ()


def test_full_preferred_span_is_its_own_tier():
    result = assess_history(_inputs(history_span_days=400), POLICY, available_count=5)
    assert result.admissibility == ADMISSIBILITY_PREFERRED
    assert result.sufficient is True


def test_rows_existing_is_not_sufficiency_short_span_fails():
    result = assess_history(_inputs(history_span_days=30), POLICY, available_count=5)
    assert result.admissibility == ADMISSIBILITY_USABLE_WEAK
    assert result.sufficient is False
    assert HISTORY_SPAN_BELOW_MINIMUM in result.reasons


def test_below_minimum_completed_work_orders_fails():
    result = assess_history(
        _inputs(completed_work_orders=45, delayed_work_orders=45), POLICY, available_count=5
    )
    assert result.admissibility == ADMISSIBILITY_USABLE_WEAK
    assert result.sufficient is False
    assert COMPLETED_WORK_ORDERS_BELOW_MINIMUM in result.reasons


def test_below_sufficiency_but_above_usable_floor_delayed_work_orders_fails():
    # For 5 signals the floor is 40; 45 is above the floor but below the
    # sufficiency bar (50): usable_weak, not inadmissible.
    result = assess_history(_inputs(delayed_work_orders=45), POLICY, available_count=5)
    assert result.admissibility == ADMISSIBILITY_USABLE_WEAK
    assert result.sufficient is False
    assert DELAYED_WORK_ORDERS_BELOW_MINIMUM in result.reasons
    assert BELOW_USABLE_FLOOR not in result.reasons


def test_at_or_below_usable_floor_is_inadmissible_not_merely_weak():
    result = assess_history(_inputs(delayed_work_orders=40), POLICY, available_count=5)  # == floor
    assert result.admissibility == ADMISSIBILITY_INADMISSIBLE
    assert result.sufficient is False
    assert BELOW_USABLE_FLOOR in result.reasons

    result_below = assess_history(_inputs(delayed_work_orders=10), POLICY, available_count=5)
    assert result_below.admissibility == ADMISSIBILITY_INADMISSIBLE


def test_same_delayed_work_orders_count_is_admissible_with_fewer_signals():
    # 15 delayed work orders is inadmissible for 5 signals (floor=40) but
    # admissible for 2 signals (floor=10) — the whole point of making the
    # floor signal-aware instead of a fixed assumption of 5.
    inputs = _inputs(delayed_work_orders=15)
    result_five = assess_history(inputs, POLICY, available_count=5)
    result_two = assess_history(inputs, POLICY, available_count=2)
    assert result_five.admissibility == ADMISSIBILITY_INADMISSIBLE
    assert result_two.admissibility != ADMISSIBILITY_INADMISSIBLE


def test_low_signal_coverage_fails():
    coverage = {signal: 0.9 for signal in SIGNAL_ORDER}
    coverage["seasonality"] = 0.1
    result = assess_history(_inputs(per_signal_coverage=coverage), POLICY, available_count=5)
    assert result.admissibility == ADMISSIBILITY_USABLE_WEAK
    assert result.sufficient is False
    assert SIGNAL_COVERAGE_BELOW_MINIMUM in result.reasons


def test_every_failing_criterion_is_recorded_simultaneously():
    result = assess_history(
        _inputs(history_span_days=1, completed_work_orders=45, delayed_work_orders=45),
        POLICY,
        available_count=5,
    )
    assert result.admissibility == ADMISSIBILITY_USABLE_WEAK
    assert result.sufficient is False
    assert HISTORY_SPAN_BELOW_MINIMUM in result.reasons
    assert COMPLETED_WORK_ORDERS_BELOW_MINIMUM in result.reasons
    assert DELAYED_WORK_ORDERS_BELOW_MINIMUM in result.reasons


def test_config_thresholds_are_read_not_hardcoded():
    lenient = HistoryPolicyConfig(min_history_span_days=1)
    result = assess_history(_inputs(history_span_days=2), lenient, available_count=5)
    assert HISTORY_SPAN_BELOW_MINIMUM not in result.reasons


def test_events_per_parameter_is_read_from_config_not_hardcoded():
    stricter = HistoryPolicyConfig(events_per_parameter=25)  # floor for 5 signals = 100
    result = assess_history(_inputs(delayed_work_orders=45), stricter, available_count=5)
    assert result.admissibility == ADMISSIBILITY_INADMISSIBLE
