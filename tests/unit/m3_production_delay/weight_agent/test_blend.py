import random

import pytest

from m3_production_delay.llm_agents.weight_agent.blend import blend_weights, compute_lambda_bp
from m3_production_delay.llm_agents.weight_agent.history_policy import compute_usable_floor
from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER, TOTAL_BP

PRIOR_BP = {
    "time_overrun": 4000,
    "operator_skill": 3500,
    "seasonality": 1000,
    "material_availability": 1000,
    "supplier_reliability": 500,
}
FITTED_BP = {
    "time_overrun": 3200,
    "operator_skill": 4100,
    "seasonality": 800,
    "material_availability": 1400,
    "supplier_reliability": 500,
}
EVENTS_PER_PARAMETER = 10
K = 40


def test_lambda_zero_returns_prior_exactly():
    result = blend_weights(PRIOR_BP, FITTED_BP, 0, SIGNAL_ORDER)
    assert result == PRIOR_BP


def test_lambda_10000_returns_fitted_exactly():
    result = blend_weights(PRIOR_BP, FITTED_BP, TOTAL_BP, SIGNAL_ORDER)
    assert result == FITTED_BP


def test_blend_always_sums_to_10000():
    rng = random.Random(99)
    for _ in range(200):
        lam = rng.randint(0, TOTAL_BP)
        result = blend_weights(PRIOR_BP, FITTED_BP, lam, SIGNAL_ORDER)
        assert sum(result.values()) == TOTAL_BP


def test_lambda_bp_is_monotonic_in_n():
    values = [compute_lambda_bp(n, K, n_floor=10) for n in range(0, 2000, 10)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_lambda_bp_zero_history_is_zero():
    assert compute_lambda_bp(0, K, n_floor=10) == 0


def test_lambda_bp_defends_against_a_negative_k_directly_called():
    # config.WeightAgentConfig.__post_init__ already forbids shrinkage_k < 0
    # — this proves the pure function itself doesn't divide by a
    # non-positive denominator if called directly, bypassing that guard.
    assert compute_lambda_bp(100, k=-200, n_floor=0) == 0


def test_lambda_bp_never_exceeds_10000():
    for n in (0, 1, 1000, 10**6, 10**12):
        assert compute_lambda_bp(n, K, n_floor=0) <= TOTAL_BP


def test_lambda_bp_stays_well_below_10000_for_realistic_n():
    # With rounding (not flooring), lambda_bp DOES reach exactly 10000 for an
    # astronomically large n relative to k — correct, since overwhelming
    # data relative to k should mean full trust. For any realistic number of
    # delayed work orders it stays comfortably below.
    assert compute_lambda_bp(10_000, K, n_floor=0) < TOTAL_BP


def test_lambda_bp_reaches_10000_only_for_extreme_n_relative_to_k():
    assert compute_lambda_bp(10**7, K, n_floor=0) == TOTAL_BP


# --- shrinkage_k and events_per_parameter must never be derived from each other ---


def test_n_floor_is_read_from_argument_not_derived_from_k():
    # same n, same k, different n_floor -> different lambda. If n_floor were
    # somehow derived from k internally, passing it explicitly would have no
    # effect, which this disproves.
    lam_low_floor = compute_lambda_bp(50, K, n_floor=5)
    lam_high_floor = compute_lambda_bp(50, K, n_floor=45)
    assert lam_low_floor > lam_high_floor


def test_changing_shrinkage_k_does_not_change_the_floor_itself():
    # compute_usable_floor never takes k as an argument at all -- this is
    # the structural guarantee that the two concepts can't be conflated.
    floor_a = compute_usable_floor(5, EVENTS_PER_PARAMETER)
    floor_b = compute_usable_floor(5, EVENTS_PER_PARAMETER)
    assert floor_a == floor_b == 40  # k plays no part in this computation at all


def test_changing_events_per_parameter_changes_the_floor():
    assert compute_usable_floor(5, events_per_parameter=5) != compute_usable_floor(5, events_per_parameter=20)


def test_changing_available_signal_count_changes_the_floor():
    assert compute_usable_floor(5, EVENTS_PER_PARAMETER) != compute_usable_floor(3, EVENTS_PER_PARAMETER)


# --- boundary behaviour at the signal-aware floor, for every signal count ---


@pytest.mark.parametrize(
    "available_count,expected_floor",
    [(5, 40), (4, 30), (3, 20), (2, 10)],
)
def test_below_at_and_above_floor_for_each_signal_count(available_count, expected_floor):
    floor = compute_usable_floor(available_count, EVENTS_PER_PARAMETER)
    assert floor == expected_floor

    if floor > 0:
        assert compute_lambda_bp(floor - 1, K, floor) == 0
    assert compute_lambda_bp(floor, K, floor) == 0  # AT the floor: still inadmissible
    lam_above = compute_lambda_bp(floor + 1, K, floor)
    assert lam_above > 0

    lam_larger = compute_lambda_bp(floor + 200, K, floor)
    assert lam_larger > lam_above
    assert lam_larger <= TOTAL_BP


def test_one_available_signal_has_no_floor_and_any_positive_n_gives_nonzero_lambda():
    # free_parameters=0 -> floor=0 -> there's no "not enough data" question
    # to ask; the resolver doesn't even need a fitted blend here since the
    # only valid weight vector is 100% on the one signal, but the pure
    # function itself must still behave sensibly if called.
    floor = compute_usable_floor(1, EVENTS_PER_PARAMETER)
    assert floor == 0
    assert compute_lambda_bp(0, K, floor) == 0  # zero observations is still zero evidence
    assert compute_lambda_bp(1, K, floor) > 0  # but any observation at all is admissible
    assert compute_lambda_bp(1000, K, floor) <= TOTAL_BP


@pytest.mark.parametrize(
    "delayed_events,expected_effective_n,expected_percent",
    [
        (20, 0, 0),
        (40, 0, 0),
        (50, 10, 20),
        (60, 20, 33),
        (80, 40, 50),
        (120, 80, 67),
        (200, 160, 80),
    ],
)
def test_expected_influence_curve_for_five_signals(delayed_events, expected_effective_n, expected_percent):
    floor = compute_usable_floor(5, EVENTS_PER_PARAMETER)  # 40
    assert floor == 40
    effective_n = max(0, delayed_events - floor)
    assert effective_n == expected_effective_n
    lam = compute_lambda_bp(delayed_events, K, floor)
    assert round(lam / 100) == expected_percent  # bp -> whole percent, rounded for the table's precision


def test_no_discontinuity_around_a_500_completed_work_order_threshold():
    # The anti-pattern the spec calls out: a naive "n>=500 ? fitted : prior"
    # branch jumps 0 -> ~10000bp right at the boundary. The shrinkage formula
    # must not: n=499 vs n=501 (straddling a typical sufficiency threshold,
    # both comfortably above the admissibility floor) should differ by only a
    # few bp, not by thousands.
    lam_499 = compute_lambda_bp(499, K, n_floor=10)
    lam_501 = compute_lambda_bp(501, K, n_floor=10)
    assert abs(lam_501 - lam_499) < 50

    blend_499 = blend_weights(PRIOR_BP, FITTED_BP, lam_499, SIGNAL_ORDER)
    blend_501 = blend_weights(PRIOR_BP, FITTED_BP, lam_501, SIGNAL_ORDER)
    for signal in SIGNAL_ORDER:
        assert abs(blend_499[signal] - blend_501[signal]) <= 5


def test_intermediate_lambda_is_between_prior_and_fitted_per_signal():
    lam = compute_lambda_bp(100, K, n_floor=10)
    assert 0 < lam < TOTAL_BP
    result = blend_weights(PRIOR_BP, FITTED_BP, lam, SIGNAL_ORDER)
    for signal in SIGNAL_ORDER:
        lo, hi = sorted((PRIOR_BP[signal], FITTED_BP[signal]))
        assert lo - 1 <= result[signal] <= hi + 1  # +/-1 tolerance for remainder rounding
