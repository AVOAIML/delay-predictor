import pytest

from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER
from m3_production_delay.llm_agents.weight_agent.rounding import assign_remainder_canonical, ensure_finite_ints


def test_zero_residual_returns_copy_unchanged():
    values = {"a": 1, "b": 2}
    result = assign_remainder_canonical(values, 0, ["a", "b"])
    assert result == values
    assert result is not values


def test_positive_residual_assigns_in_canonical_order():
    values = {s: 0 for s in SIGNAL_ORDER}
    result = assign_remainder_canonical(values, 3, SIGNAL_ORDER)
    # first three signals in canonical order get +1 each
    assert result[SIGNAL_ORDER[0]] == 1
    assert result[SIGNAL_ORDER[1]] == 1
    assert result[SIGNAL_ORDER[2]] == 1
    assert result[SIGNAL_ORDER[3]] == 0
    assert result[SIGNAL_ORDER[4]] == 0
    assert sum(result.values()) == 3


def test_negative_residual_subtracts_in_canonical_order():
    values = {s: 10 for s in SIGNAL_ORDER}
    result = assign_remainder_canonical(values, -2, SIGNAL_ORDER)
    assert result[SIGNAL_ORDER[0]] == 9
    assert result[SIGNAL_ORDER[1]] == 9
    assert result[SIGNAL_ORDER[2]] == 10
    assert sum(result.values()) == 48


def test_wraps_around_order_when_residual_exceeds_signal_count():
    values = {s: 0 for s in SIGNAL_ORDER}
    residual = len(SIGNAL_ORDER) + 2
    result = assign_remainder_canonical(values, residual, SIGNAL_ORDER)
    assert sum(result.values()) == residual
    # first two signals get an extra unit from the second lap
    assert result[SIGNAL_ORDER[0]] == 2
    assert result[SIGNAL_ORDER[1]] == 2
    assert result[SIGNAL_ORDER[2]] == 1


def test_is_deterministic_across_repeated_calls():
    values = {s: 5 for s in SIGNAL_ORDER}
    first = assign_remainder_canonical(values, 7, SIGNAL_ORDER)
    second = assign_remainder_canonical(values, 7, SIGNAL_ORDER)
    assert first == second


def test_capacity_skips_saturated_signals():
    values = {"a": 0, "b": 0, "c": 0}
    capacity = {"a": 0, "b": 1, "c": 5}
    result = assign_remainder_canonical(values, 3, ["a", "b", "c"], capacity=capacity)
    assert result["a"] == 0
    assert result["b"] == 1
    assert result["c"] == 2
    assert sum(result.values()) == 3


def test_capacity_insufficient_raises():
    values = {"a": 0, "b": 0}
    capacity = {"a": 0, "b": 1}
    try:
        assign_remainder_canonical(values, 5, ["a", "b"], capacity=capacity)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for insufficient capacity")


def test_empty_order_with_nonzero_residual_raises():
    try:
        assign_remainder_canonical({}, 1, [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty order")


# --- Improvement 6: no NaN/Infinity, no non-integer, accepted -------------


def test_ensure_finite_ints_rejects_float():
    with pytest.raises(ValueError):
        ensure_finite_ints({"a": 1.5}, "ctx")


def test_ensure_finite_ints_rejects_bool():
    with pytest.raises(ValueError):
        ensure_finite_ints({"a": True}, "ctx")


def test_ensure_finite_ints_rejects_nan():
    with pytest.raises(ValueError):
        ensure_finite_ints({"a": float("nan")}, "ctx")


def test_ensure_finite_ints_accepts_plain_ints():
    ensure_finite_ints({"a": -5, "b": 0, "c": 10_000}, "ctx")  # must not raise


def test_assign_remainder_canonical_rejects_non_integer_values():
    with pytest.raises(ValueError):
        assign_remainder_canonical({"a": 1.5}, 1, ["a"])


def test_assign_remainder_canonical_rejects_non_integer_residual():
    with pytest.raises(ValueError):
        assign_remainder_canonical({"a": 0}, 1.5, ["a"])


def test_assign_remainder_canonical_rejects_bool_residual():
    with pytest.raises(ValueError):
        assign_remainder_canonical({"a": 0}, True, ["a"])


def test_assign_remainder_canonical_rejects_non_integer_capacity():
    with pytest.raises(ValueError):
        assign_remainder_canonical({"a": 0}, 1, ["a"], capacity={"a": 1.5})
