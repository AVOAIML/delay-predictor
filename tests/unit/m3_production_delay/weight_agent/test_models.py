"""Improvement 3: every domain object here must validate itself at
construction — these tests exist to prove that claim, not to pad coverage.
Each one constructs an object no caller should be able to build and confirms
it is rejected at __post_init__, not merely "usually checked by the caller".
"""

import pytest

from m3_production_delay.llm_agents.weight_agent.models import (
    ADMISSIBILITY_INADMISSIBLE,
    ADMISSIBILITY_SUFFICIENT,
    SIGNAL_ORDER,
    TOTAL_BP,
    Bounds,
    ExcludedSignal,
    FittedWeights,
    HistoryAssessment,
    HistoryInputs,
    TenantProfile,
    WeightResolution,
    bp_to_percent,
)

VALID_WEIGHTS = {
    "time_overrun": 4000,
    "operator_skill": 3500,
    "seasonality": 1000,
    "material_availability": 1000,
    "supplier_reliability": 500,
}
VALID_BOUNDS = {
    "time_overrun": Bounds(3000, 5000),
    "operator_skill": Bounds(2000, 4500),
    "seasonality": Bounds(500, 2000),
    "material_availability": Bounds(500, 2500),
    "supplier_reliability": Bounds(300, 1500),
}


def _valid_resolution_kwargs(**overrides):
    kwargs = dict(
        tenant_id="tenant_a",
        source="prior",
        status="recommendation",
        weights_bp=dict(VALID_WEIGHTS),
        available_signals=SIGNAL_ORDER,
        excluded_signals=(),
        history_assessment=HistoryAssessment(admissibility=ADMISSIBILITY_INADMISSIBLE),
        prior_bp=dict(VALID_WEIGHTS),
        adjustments_bp={s: 0 for s in SIGNAL_ORDER},
        bounds_bp=dict(VALID_BOUNDS),
        confidence=0.0,
        evidence=(),
        fallback_reasons=(),
        requires_admin_approval=True,
        prompt_version="v1",
        transform_version="v1",
        agent_version="v1",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    kwargs.update(overrides)
    return kwargs


# --- Bounds -----------------------------------------------------------------


def test_bounds_rejects_inverted_range():
    with pytest.raises(ValueError):
        Bounds(200, 100)


def test_bounds_rejects_negative_min():
    with pytest.raises(ValueError):
        Bounds(-1, 100)


def test_bounds_rejects_max_over_total():
    with pytest.raises(ValueError):
        Bounds(0, 10_001)


def test_bounds_accepts_equal_min_max():
    b = Bounds(500, 500)
    assert b.contains(500)
    assert not b.contains(501)


def test_bp_to_percent_is_derived_and_lossy():
    assert bp_to_percent({"a": 250}) == {"a": 2.5}


# --- ExcludedSignal -----------------------------------------------------


def test_excluded_signal_rejects_unknown_signal_name():
    with pytest.raises(ValueError):
        ExcludedSignal("not_a_real_signal", "reason")


def test_excluded_signal_rejects_empty_reason():
    with pytest.raises(ValueError):
        ExcludedSignal("seasonality", "")


# --- HistoryAssessment ----------------------------------------------------


def test_history_assessment_rejects_unknown_admissibility():
    with pytest.raises(ValueError):
        HistoryAssessment(admissibility="somewhere_in_between")  # type: ignore[arg-type]


def test_history_assessment_rejects_lambda_out_of_range():
    with pytest.raises(ValueError):
        HistoryAssessment(admissibility=ADMISSIBILITY_SUFFICIENT, lambda_bp=10_001)
    with pytest.raises(ValueError):
        HistoryAssessment(admissibility=ADMISSIBILITY_SUFFICIENT, lambda_bp=-1)


def test_history_assessment_allows_inadmissible_tenant_posture_with_a_trusted_fit():
    # Deliberately legitimate since the fitted-provenance design: admissibility
    # reports the TENANT's own resolution-time history; lambda_bp can come
    # from a fitted artifact's independent, sufficient evidence instead.
    ha = HistoryAssessment(admissibility=ADMISSIBILITY_INADMISSIBLE, lambda_bp=100)
    assert ha.lambda_bp == 100
    assert ha.sufficient is False


def test_history_assessment_sufficient_property():
    assert HistoryAssessment(admissibility=ADMISSIBILITY_SUFFICIENT).sufficient is True
    assert HistoryAssessment(admissibility=ADMISSIBILITY_INADMISSIBLE).sufficient is False


# --- HistoryInputs --------------------------------------------------------


def test_history_inputs_rejects_negative_counts():
    with pytest.raises(ValueError):
        HistoryInputs(history_span_days=-1, completed_work_orders=1, delayed_work_orders=0)


def test_history_inputs_rejects_delayed_exceeding_completed():
    with pytest.raises(ValueError):
        HistoryInputs(history_span_days=10, completed_work_orders=5, delayed_work_orders=6)


def test_history_inputs_rejects_unknown_signal_in_coverage():
    with pytest.raises(ValueError):
        HistoryInputs(
            history_span_days=10,
            completed_work_orders=5,
            delayed_work_orders=1,
            per_signal_coverage={"not_a_signal": 0.5},
        )


def test_history_inputs_rejects_coverage_out_of_range():
    with pytest.raises(ValueError):
        HistoryInputs(
            history_span_days=10,
            completed_work_orders=5,
            delayed_work_orders=1,
            per_signal_coverage={"time_overrun": 1.5},
        )


# --- FittedWeights ---------------------------------------------------------
# Contract: weights_bp is always the complete five-signal mapping; signals
# outside signal_set must be exactly 0; signals inside signal_set must sum
# to exactly 10000; delayed_event_count >= 0.

FULL_SIGNAL_SET = frozenset(SIGNAL_ORDER)


def _fitted_weights_over(signal_set, delayed_event_count=100):
    per_signal = TOTAL_BP // len(signal_set)
    weights = {s: 0 for s in SIGNAL_ORDER}
    remainder = TOTAL_BP - per_signal * len(signal_set)
    for i, signal in enumerate(sorted(signal_set)):
        weights[signal] = per_signal + (1 if i < remainder else 0)
    return FittedWeights(
        weights_bp=weights, signal_set=frozenset(signal_set), delayed_event_count=delayed_event_count
    )


def test_fitted_weights_accepts_a_valid_full_signal_set():
    fw = _fitted_weights_over(FULL_SIGNAL_SET)
    assert fw.signal_set == FULL_SIGNAL_SET
    assert sum(fw.weights_bp.values()) == TOTAL_BP


def test_fitted_weights_accepts_a_valid_partial_signal_set_with_zeros_outside():
    partial = frozenset({"time_overrun", "operator_skill"})
    fw = _fitted_weights_over(partial)
    assert sum(fw.weights_bp[s] for s in partial) == TOTAL_BP
    for signal in FULL_SIGNAL_SET - partial:
        assert fw.weights_bp[signal] == 0


def test_fitted_weights_rejects_empty_signal_set():
    with pytest.raises(Exception):
        FittedWeights(
            weights_bp={s: 0 for s in SIGNAL_ORDER}, signal_set=frozenset(), delayed_event_count=10
        )


def test_fitted_weights_rejects_unknown_signal_in_signal_set():
    with pytest.raises(Exception):
        FittedWeights(
            weights_bp={s: 0 for s in SIGNAL_ORDER},
            signal_set=frozenset({"not_a_real_signal"}),
            delayed_event_count=10,
        )


def test_fitted_weights_rejects_wrong_weights_bp_signal_set():
    with pytest.raises(Exception):
        FittedWeights(
            weights_bp={"time_overrun": TOTAL_BP},  # missing the other 4 keys
            signal_set=frozenset({"time_overrun"}),
            delayed_event_count=10,
        )


def test_fitted_weights_rejects_negative_value():
    bad = {s: 0 for s in SIGNAL_ORDER}
    bad["time_overrun"] = TOTAL_BP + 1
    bad["operator_skill"] = -1  # sums correctly overall, but a negative value alone must reject
    with pytest.raises(Exception):
        FittedWeights(weights_bp=bad, signal_set=FULL_SIGNAL_SET, delayed_event_count=10)


def test_fitted_weights_rejects_float_value():
    bad = {s: 0 for s in SIGNAL_ORDER}
    bad["time_overrun"] = 100.5
    with pytest.raises(Exception):
        FittedWeights(weights_bp=bad, signal_set=FULL_SIGNAL_SET, delayed_event_count=10)


def test_fitted_weights_rejects_nonzero_weight_outside_signal_set():
    weights = {s: 0 for s in SIGNAL_ORDER}
    weights["time_overrun"] = TOTAL_BP
    weights["seasonality"] = 1  # outside the declared signal_set below
    with pytest.raises(Exception):
        FittedWeights(
            weights_bp=weights,
            signal_set=frozenset({"time_overrun"}),
            delayed_event_count=10,
        )


def test_fitted_weights_rejects_signal_set_weights_not_summing_to_10000():
    weights = {s: 0 for s in SIGNAL_ORDER}
    weights["time_overrun"] = 5000  # signal_set is just this one signal -> must be 10000
    with pytest.raises(Exception):
        FittedWeights(
            weights_bp=weights,
            signal_set=frozenset({"time_overrun"}),
            delayed_event_count=10,
        )


def test_fitted_weights_rejects_negative_delayed_event_count():
    with pytest.raises(Exception):
        _fitted_weights_over(FULL_SIGNAL_SET, delayed_event_count=-1)


def test_fitted_weights_accepts_zero_delayed_event_count():
    fw = _fitted_weights_over(FULL_SIGNAL_SET, delayed_event_count=0)
    assert fw.delayed_event_count == 0


def test_fitted_weights_single_signal_forces_10000_deterministically():
    fw = _fitted_weights_over(frozenset({"supplier_reliability"}))
    assert fw.weights_bp["supplier_reliability"] == TOTAL_BP
    assert all(fw.weights_bp[s] == 0 for s in FULL_SIGNAL_SET - {"supplier_reliability"})


# --- TenantProfile --------------------------------------------------------


def test_tenant_profile_rejects_out_of_enum_value():
    with pytest.raises(ValueError):
        TenantProfile(industry="not_a_real_industry")


def test_tenant_profile_accepts_all_null():
    profile = TenantProfile()
    assert set(profile.missing_critical_fields()) == {
        "production_type",
        "material_dependency",
        "supplier_dependency",
        "workforce_dependency",
        "seasonality_level",
    }


def test_tenant_profile_missing_critical_fields_order_is_deterministic():
    a = TenantProfile().missing_critical_fields()
    b = TenantProfile().missing_critical_fields()
    assert a == b


def test_tenant_profile_missing_critical_fields_empty_when_all_present():
    profile = TenantProfile(
        production_type="make_to_order",
        material_dependency="high",
        supplier_dependency="high",
        workforce_dependency="medium",
        seasonality_level="low",
    )
    assert profile.missing_critical_fields() == ()


# --- WeightResolution ------------------------------------------------------


def test_weight_resolution_rejects_unknown_source():
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(source="configureed"))


def test_weight_resolution_rejects_unknown_status():
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(status="recomendation"))


def test_weight_resolution_rejects_active_status_for_non_configured_source():
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(source="prior", status="active"))


def test_weight_resolution_rejects_admin_approval_true_for_configured():
    with pytest.raises(ValueError):
        WeightResolution(
            **_valid_resolution_kwargs(
                source="configured", status="active", requires_admin_approval=True
            )
        )


def test_weight_resolution_rejects_admin_approval_false_for_non_configured():
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(requires_admin_approval=False))


def test_weight_resolution_rejects_confidence_out_of_range():
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(confidence=1.5))


def test_weight_resolution_rejects_confidence_nan():
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(confidence=float("nan")))


def test_weight_resolution_rejects_weights_not_summing_to_10000():
    bad_weights = dict(VALID_WEIGHTS)
    bad_weights["time_overrun"] += 1
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(weights_bp=bad_weights))


def test_weight_resolution_rejects_missing_signal_in_weights():
    incomplete = dict(VALID_WEIGHTS)
    del incomplete["seasonality"]
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(weights_bp=incomplete))


def test_weight_resolution_rejects_adjustments_not_summing_to_zero():
    bad_adjustments = {s: 0 for s in SIGNAL_ORDER}
    bad_adjustments["time_overrun"] = 5
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(adjustments_bp=bad_adjustments))


def test_weight_resolution_rejects_weight_outside_its_own_bound():
    bad_weights = dict(VALID_WEIGHTS)
    bad_weights["time_overrun"] = 100  # below its bound's min of 3000
    bad_weights["operator_skill"] = 7400  # compensate to keep sum at 10000
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(weights_bp=bad_weights))


def test_weight_resolution_rejects_all_signals_unavailable_zero_sum_state():
    # Blocker fix: there is no valid all-zero WeightResolution. Zero
    # available signals means no vector can be constructed at all — the
    # resolver must raise AllSignalsUnavailableError before ever reaching
    # this constructor (see test_resolver.py); this proves the type itself
    # refuses the state too, not just the resolver's call site.
    zero = {s: 0 for s in SIGNAL_ORDER}
    zero_bounds = {s: Bounds(0, 0) for s in SIGNAL_ORDER}
    with pytest.raises(ValueError):
        WeightResolution(
            **_valid_resolution_kwargs(
                weights_bp=zero,
                prior_bp=zero,
                bounds_bp=zero_bounds,
                available_signals=(),
                excluded_signals=tuple(
                    ExcludedSignal(s, "all_signals_unavailable") for s in SIGNAL_ORDER
                ),
            )
        )


def test_weight_resolution_rejects_empty_tenant_id():
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(tenant_id=""))


def test_weight_resolution_rejects_fitted_projection_applied_for_non_fitted_source():
    with pytest.raises(ValueError):
        WeightResolution(**_valid_resolution_kwargs(source="prior", fitted_projection_applied=True))


def test_weight_resolution_rejects_changed_signals_when_projection_not_applied():
    with pytest.raises(ValueError):
        WeightResolution(
            **_valid_resolution_kwargs(
                source="blended",
                requires_admin_approval=True,
                fitted_projection_applied=False,
                fitted_projection_changed_signals=("time_overrun",),
            )
        )


def test_weight_resolution_rejects_unknown_signal_in_changed_signals():
    with pytest.raises(ValueError):
        WeightResolution(
            **_valid_resolution_kwargs(
                source="blended",
                requires_admin_approval=True,
                fitted_projection_applied=True,
                fitted_projection_changed_signals=("not_a_real_signal",),
            )
        )


def test_weight_resolution_accepts_valid_fitted_projection_metadata():
    result = WeightResolution(
        **_valid_resolution_kwargs(
            source="blended",
            requires_admin_approval=True,
            fitted_projection_applied=True,
            fitted_projection_changed_signals=("time_overrun", "seasonality"),
        )
    )
    assert result.fitted_projection_applied is True
    assert result.fitted_projection_changed_signals == ("time_overrun", "seasonality")
    assert result.to_json_dict()["fitted_projection_applied"] is True
