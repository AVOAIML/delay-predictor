import pytest

from m3_production_delay.llm_agents.weight_agent.config import (
    DEFAULT_BOUNDS_BP,
    DEFAULT_PRIOR_BP,
    DOMAIN_PRIOR_BP,
    HistoryPolicyConfig,
    WeightAgentConfig,
    build_config,
    get_weight_agent_config,
)
from m3_production_delay.llm_agents.weight_agent.exceptions import WeightConfigError
from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER, TOTAL_BP, Bounds


def test_default_config_loads_and_validates():
    config = build_config()
    assert sum(config.prior_bp.values()) == TOTAL_BP
    for signal, bound in config.bounds_bp.items():
        assert bound.contains(config.prior_bp[signal])


def test_cached_singleton_is_stable_until_cleared():
    get_weight_agent_config.cache_clear()
    first = get_weight_agent_config()
    second = get_weight_agent_config()
    assert first is second


def test_prior_not_summing_to_10000_fails_at_load():
    bad_prior = dict(DEFAULT_PRIOR_BP)
    bad_prior["time_overrun"] += 1  # now sums to 10001
    with pytest.raises(WeightConfigError):
        build_config(prior_bp=bad_prior)


def test_infeasible_bounds_fail_at_load():
    # sum(max) forced below 10000 -> no vector can ever sum to TOTAL_BP
    bad_bounds = dict(DEFAULT_BOUNDS_BP)
    bad_bounds["supplier_reliability"] = Bounds(300, 400)
    bad_bounds["material_availability"] = Bounds(500, 600)
    with pytest.raises(WeightConfigError):
        build_config(bounds_bp=bad_bounds)


def test_prior_outside_its_own_bound_fails_at_load():
    bad_bounds = dict(DEFAULT_BOUNDS_BP)
    bad_bounds["time_overrun"] = Bounds(4500, 5000)  # prior is 4000, now out of range
    with pytest.raises(WeightConfigError):
        build_config(bounds_bp=bad_bounds)


def test_missing_signal_in_prior_fails_at_load():
    incomplete = dict(DEFAULT_PRIOR_BP)
    del incomplete["seasonality"]
    with pytest.raises(WeightConfigError):
        build_config(prior_bp=incomplete)


def test_invalid_max_projection_iterations_fails_at_load():
    with pytest.raises(WeightConfigError):
        build_config(max_projection_iterations=0)


def test_invalid_min_profile_confidence_fails_at_load():
    with pytest.raises(WeightConfigError):
        build_config(min_profile_confidence=1.5)


# --- Improvement 3: self-validating dataclass, not just build_config -------


def test_direct_construction_bypassing_build_config_still_validates():
    bad_prior = dict(DEFAULT_PRIOR_BP)
    bad_prior["time_overrun"] += 1
    with pytest.raises(WeightConfigError):
        WeightAgentConfig(
            prior_bp=bad_prior,
            bounds_bp=DEFAULT_BOUNDS_BP,
            max_projection_iterations=20,
            history_policy=HistoryPolicyConfig(),
            shrinkage_k=40,
            llm_enabled=False,
        )


def test_history_policy_config_validates_itself_directly():
    with pytest.raises(WeightConfigError):
        HistoryPolicyConfig(min_signal_coverage=1.5)
    with pytest.raises(WeightConfigError):
        HistoryPolicyConfig(min_history_span_days=-1)


# --- events_per_parameter is independent of shrinkage_k (not derived from it) ---


def test_events_per_parameter_defaults_to_ten():
    assert HistoryPolicyConfig().events_per_parameter == 10


def test_changing_shrinkage_k_does_not_change_events_per_parameter():
    config_k40 = build_config(shrinkage_k=40)
    config_k80 = build_config(shrinkage_k=80)
    assert config_k40.history_policy.events_per_parameter == config_k80.history_policy.events_per_parameter


def test_events_per_parameter_must_be_at_least_one():
    with pytest.raises(WeightConfigError):
        HistoryPolicyConfig(events_per_parameter=0)
    with pytest.raises(WeightConfigError):
        HistoryPolicyConfig(events_per_parameter=-1)


def test_explicit_history_policy_is_used_verbatim_regardless_of_shrinkage_k():
    explicit = HistoryPolicyConfig(events_per_parameter=3)
    config = build_config(shrinkage_k=80, history_policy=explicit)
    assert config.history_policy.events_per_parameter == 3


# --- domain prior is a product decision, not a statistical fit -------------


def test_domain_prior_is_the_shipped_default():
    assert DEFAULT_PRIOR_BP == DOMAIN_PRIOR_BP
    assert sum(DOMAIN_PRIOR_BP.values()) == TOTAL_BP
    assert set(DOMAIN_PRIOR_BP) == set(SIGNAL_ORDER)
