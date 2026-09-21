"""Evidence builder: the four things Section 3 derives that the Risk Engine
does not expose, plus the coercions that keep a payload serialisable."""

from __future__ import annotations

import json
import math

import pytest

from conftest import make_component, make_history_entry, make_op, make_operator, make_po, make_vendor
from m3_production_delay.review.evidence import (
    FIRE_BASELINES,
    SCORING_GATE_MIN_TIME_RATIO,
    build_evidence,
    fired,
    is_scorable,
    operation_label,
    overrun_basis,
    supplier_reliability,
)
from m3_production_delay.review.schemas import (
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_PREDECESSOR_OVERRUN,
    SIGNAL_SUPPLIER_RELIABILITY,
    SIGNAL_TIME_OVERRUN,
)
from maxxflow_core.jsonutil import json_default


def _job(ops: list[dict], job_id: str = "WH/MO/00999") -> dict:
    return {"job_id": job_id, "operations": ops}


# ─── supplier_reliability (recomputed, because Section 1 does not expose it) ──


def test_supplier_reliability_averages_vendor_ratios():
    components = [
        {"vendor": {"vendor_lead_time_ratio": 1.5}},
        {"vendor": {"vendor_lead_time_ratio": 0.5}},
    ]
    assert supplier_reliability(components) == pytest.approx(1.0)


def test_supplier_reliability_excludes_components_without_a_usable_vendor():
    components = [
        {"vendor": {"vendor_lead_time_ratio": 2.0}},
        {"vendor": None},
        {"vendor": {"vendor_lead_time_ratio": None}},
        {},
    ]
    # Excluded, not counted as an on-time 1.0 — otherwise three components
    # with no vendor history would drag a genuinely late supplier to 1.25.
    assert supplier_reliability(components) == pytest.approx(2.0)


def test_supplier_reliability_is_none_without_any_vendor_ratio():
    assert supplier_reliability([{"vendor": None}]) is None


def test_supplier_reliability_matches_the_engine_on_a_scored_job(weights, threshold):
    vendor = make_vendor([make_po("2026-08-01", "2026-08-15", "2026-08-22")])
    op = make_op(components=[make_component(100, 40, vendor)])
    # The engine folded this number into composite_risk_score without ever
    # attaching it; rebuilding it from the ratios it DID attach is the whole
    # point of the recomputation.
    op["components"][0]["vendor"]["vendor_lead_time_ratio"] = 1.5
    op.update({"composite_risk_score": 1.2, "is_delayed": True, "time_overrun_ratio": 1.1,
               "operator_pace_ratio": None, "material_shortfall_ratio": 2.5,
               "predecessor_time_overrun_ratio": None, "predicted_overrun_hours": 1.0})
    pack = build_evidence(_job([op]), weights, threshold)
    assert pack.operations[0].signal(SIGNAL_SUPPLIER_RELIABILITY).value == pytest.approx(1.5)


# ─── overrun_basis ───────────────────────────────────────────────────────────


def test_overrun_basis_quantity_when_there_is_progress_to_extrapolate():
    assert overrun_basis(make_op(actual_duration_minutes=350, current_done_quantity=60.0)) == "quantity"


def test_overrun_basis_falls_back_to_operator_pace_without_progress():
    op = make_op(actual_duration_minutes=None, current_done_quantity=0.0, operator_pace_ratio=1.25)
    assert overrun_basis(op) == "operator_pace"


def test_overrun_basis_is_none_when_neither_is_available():
    op = make_op(actual_duration_minutes=None, current_done_quantity=0.0, operator_pace_ratio=None)
    assert overrun_basis(op) == "none"


# ─── the scoring gate Section 1 does not apply ───────────────────────────────


def test_is_scorable_requires_logged_time_and_25_percent_progress():
    assert is_scorable(make_op(actual_duration_minutes=200, time_overrun_ratio=0.5))
    assert not is_scorable(make_op(actual_duration_minutes=None, time_overrun_ratio=None))
    assert not is_scorable(make_op(actual_duration_minutes=10, time_overrun_ratio=0.2))


def test_is_scorable_at_exactly_the_gate():
    op = make_op(actual_duration_minutes=120, time_overrun_ratio=SCORING_GATE_MIN_TIME_RATIO)
    assert is_scorable(op)


# ─── labels: never invented ──────────────────────────────────────────────────


def test_operation_label_prefers_a_real_name_when_the_rollup_carries_one():
    assert operation_label(make_op(operation_name="Cutting")) == "Cutting"
    assert operation_label(make_op(work_center_name="Cutting Bay")) == "Cutting Bay"


def test_operation_label_falls_back_to_type_and_short_id():
    label = operation_label(make_op(operation_id="wo-cutting-0001", operation_type="INDEPENDENT"))
    assert label == "INDEPENDENT · wo-cutti"


# ─── fire baselines ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        (SIGNAL_TIME_OVERRUN, 1.01, True),
        (SIGNAL_TIME_OVERRUN, 1.0, False),
        (SIGNAL_OPERATOR_PACE, 1.25, True),
        (SIGNAL_OPERATOR_PACE, 1.2, False),
        (SIGNAL_MATERIAL_SHORTFALL, 0.01, True),
        (SIGNAL_MATERIAL_SHORTFALL, 0.0, False),
        (SIGNAL_SUPPLIER_RELIABILITY, 1.5, True),
        (SIGNAL_SUPPLIER_RELIABILITY, 1.0, False),
        (SIGNAL_PREDECESSOR_OVERRUN, 1.25, True),
        (SIGNAL_PREDECESSOR_OVERRUN, 1.2, False),
    ],
)
def test_fire_baselines_are_strictly_greater_than(key, value, expected):
    assert fired(key, value) is expected


def test_a_signal_with_no_value_never_fires():
    assert all(fired(key, None) is False for key in FIRE_BASELINES)


# ─── contribution shares ─────────────────────────────────────────────────────


def test_contribution_shares_sum_to_one_for_every_scorable_operation(
    section1_job, weights, threshold
):
    pack = build_evidence(section1_job, weights, threshold)
    assert pack.scorable_operations
    for op in pack.scorable_operations:
        shares = [s.contribution for s in op.signals if s.contribution is not None]
        assert shares, f"{op.operation_id} has no computable contribution"
        assert sum(shares) == pytest.approx(1.0)


def test_contribution_is_corrected_for_the_composite_renormalisation(weights, threshold):
    # Only two of the four weighted signals have a value, so the composite
    # renormalised over 0.40 + 0.30 = 0.70 of a weight vector that itself sums
    # to 0.90. A naive weight*value/score would sum to 0.70, not 1.
    op = make_op(
        time_overrun_ratio=2.0,
        operator_pace_ratio=1.5,
        material_shortfall_ratio=None,
        predecessor_time_overrun_ratio=None,
        composite_risk_score=(0.40 * 2.0 + 0.30 * 1.5) / 0.70,
        is_delayed=True,
        predicted_overrun_hours=3.0,
    )
    pack = build_evidence(_job([op]), weights, threshold)
    shares = {s.key: s.contribution for s in pack.operations[0].signals if s.contribution}
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares[SIGNAL_TIME_OVERRUN] == pytest.approx(0.8 / 1.25)


def test_contribution_is_none_rather_than_zero_when_not_computable(weights, threshold):
    op = make_op(
        time_overrun_ratio=None, operator_pace_ratio=None, material_shortfall_ratio=0.0,
        predecessor_time_overrun_ratio=None, composite_risk_score=None, is_delayed=None,
        predicted_overrun_hours=None,
    )
    pack = build_evidence(_job([op]), weights, threshold)
    assert all(s.contribution is None for s in pack.operations[0].signals)


def test_the_unweighted_predecessor_signal_never_gets_a_contribution(
    section1_job, weights, threshold
):
    pack = build_evidence(section1_job, weights, threshold)
    for op in pack.operations:
        predecessor = op.signal(SIGNAL_PREDECESSOR_OVERRUN)
        assert predecessor.weight == 0.0
        assert predecessor.contribution is None


def test_a_stray_predecessor_weight_cannot_promote_it_to_a_cause(threshold):
    # Pinned to 0 in the evidence, not read from the tenant vector.
    weights = {"time_overrun_ratio": 0.5, "predecessor_time_overrun_ratio": 0.5}
    op = make_op(
        time_overrun_ratio=1.5, operator_pace_ratio=None, material_shortfall_ratio=0.0,
        predecessor_time_overrun_ratio=2.0, composite_risk_score=1.5, is_delayed=True,
        predicted_overrun_hours=1.0,
    )
    pack = build_evidence(_job([op]), weights, threshold)
    assert pack.operations[0].signal(SIGNAL_PREDECESSOR_OVERRUN).weight == 0.0


# ─── math.inf from a zero-stock component ────────────────────────────────────


def test_infinite_shortfall_is_coerced_to_none_and_flagged(weights, threshold):
    op = make_op(
        components=[make_component(100, 0)],
        time_overrun_ratio=1.5,
        operator_pace_ratio=None,
        material_shortfall_ratio=math.inf,
        predecessor_time_overrun_ratio=None,
        composite_risk_score=math.inf,
        is_delayed=True,
        predicted_overrun_hours=2.0,
    )
    pack = build_evidence(_job([op]), weights, threshold)

    assert pack.operations[0].signal(SIGNAL_MATERIAL_SHORTFALL).value is None
    assert pack.operations[0].composite_risk_score is None
    non_finite = [i for i in pack.issues if i.check == "non_finite"]
    assert len(non_finite) >= 2
    assert all(i.severity == "warning" for i in non_finite)


def test_no_infinity_can_reach_a_serialised_payload(weights, threshold):
    op = make_op(
        components=[make_component(100, 0)],
        time_overrun_ratio=1.5, operator_pace_ratio=None, material_shortfall_ratio=math.inf,
        predecessor_time_overrun_ratio=None, composite_risk_score=math.inf, is_delayed=True,
        predicted_overrun_hours=1.0,
    )
    pack = build_evidence(_job([op]), weights, threshold)
    # allow_nan=False makes a leaked Infinity a ValueError rather than a
    # literal no JSON reader can parse.
    encoded = json.dumps(pack.to_dict(), default=json_default, allow_nan=False)
    assert "Infinity" not in encoded


def test_a_zero_stock_component_still_reaches_the_material_list(weights, threshold):
    op = make_op(
        components=[make_component(100, 0)],
        time_overrun_ratio=1.5, operator_pace_ratio=None, material_shortfall_ratio=math.inf,
        predecessor_time_overrun_ratio=None, composite_risk_score=None, is_delayed=None,
        predicted_overrun_hours=1.0,
    )
    pack = build_evidence(_job([op]), weights, threshold)
    assert [c.shortfall_quantity for c in pack.material_overrun] == [100.0]


# ─── MO-level components collapse to one job-level list ──────────────────────


def test_components_duplicated_across_operations_are_de_duplicated(
    section1_job, weights, threshold
):
    pack = build_evidence(section1_job, weights, threshold)
    # rollup.py attaches the MO's whole component list to all three operations.
    assert len(section1_job["operations"]) == 3
    assert all(len(op["components"]) == 2 for op in section1_job["operations"])
    assert [c.component_id for c in pack.material_overrun] == ["item-steel-plate"]


def test_job_scoped_material_and_supplier_evidence_is_built(section1_job, weights, threshold):
    pack = build_evidence(section1_job, weights, threshold)
    assert {s.key for s in pack.job_signals} == {
        SIGNAL_MATERIAL_SHORTFALL,
        SIGNAL_SUPPLIER_RELIABILITY,
    }
    material = pack.job_signal(SIGNAL_MATERIAL_SHORTFALL)
    assert material.value == pytest.approx(2.5)
    assert material.fired is True


# ─── job summary ─────────────────────────────────────────────────────────────


def test_summary_is_the_worst_scorable_operation_not_a_sum(section1_job, weights, threshold):
    pack = build_evidence(section1_job, weights, threshold)
    scorable = pack.scorable_operations
    assert [op.operation_id for op in scorable] == ["wo-cutting-0001", "wo-welding-0002"]
    assert pack.summary_risk_score == pytest.approx(
        max(op.composite_risk_score for op in scorable)
    )
    assert pack.summary_overrun_hours == pytest.approx(
        max(op.predicted_overrun_hours for op in scorable)
    )
    assert pack.summary_is_delayed is True
    assert pack.summary_basis == "worst_operation"


def test_summary_ignores_the_unscorable_operation_even_when_it_is_flagged_delayed(
    section1_job, weights, threshold
):
    pack = build_evidence(section1_job, weights, threshold)
    assembly = pack.operation("wo-assembly-0003")
    assert assembly.is_scorable is False
    assert assembly.is_delayed is True
    # 1.61 is the highest score in the job, and it is deliberately not the
    # summary: nothing has been logged against that operation yet.
    assert pack.summary_risk_score < assembly.composite_risk_score


def test_summary_is_delayed_always_agrees_with_score_against_threshold(
    section1_job, weights, threshold
):
    pack = build_evidence(section1_job, weights, threshold)
    assert pack.summary_is_delayed == (pack.summary_risk_score > pack.delay_threshold)


def test_summary_is_none_when_no_operation_is_scorable(weights, threshold):
    op = make_op(
        actual_duration_minutes=None, time_overrun_ratio=None, operator_pace_ratio=1.25,
        material_shortfall_ratio=0.0, predecessor_time_overrun_ratio=None,
        composite_risk_score=1.25, is_delayed=True, predicted_overrun_hours=1.0,
    )
    pack = build_evidence(_job([op]), weights, threshold)
    assert pack.scorable_operations == ()
    assert pack.summary_risk_score is None
    assert pack.summary_is_delayed is None


# ─── guards ──────────────────────────────────────────────────────────────────


def test_an_unscored_rollup_is_rejected_with_a_useful_message(weights, threshold):
    with pytest.raises(ValueError, match="calculate_delay_elements_for_jobs"):
        build_evidence(_job([make_op()]), weights, threshold)


def test_threshold_has_no_default(section1_job, weights):
    with pytest.raises(TypeError):
        build_evidence(section1_job, weights)  # type: ignore[call-arg]


def test_operators_stay_pseudonymous(section1_job, weights, threshold):
    pack = build_evidence(section1_job, weights, threshold)
    encoded = json.dumps(pack.to_dict(), default=json_default, allow_nan=False)
    # Nothing in the evidence resolves or carries an operator identity: the
    # rollup only ever holds an HMAC token, and no line needs one.
    assert "hmac-operator" not in encoded
    assert "operator_id" not in encoded


def test_history_builder_is_unused_but_shape_compatible():
    # Guards the conftest import of the rule-engine builders: if their shape
    # changes, these tests should fail here rather than somewhere confusing.
    entry = make_history_entry(500, 400)
    assert set(entry) >= {"elapsed_time_minutes", "scheduled_time_minutes", "completed_on"}
    assert make_operator([entry])["name"] is None
