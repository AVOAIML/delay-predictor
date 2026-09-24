"""Validators: one bad line (or one bad summary) per check.

Every case builds a genuinely clean draft from the recorded Section 1 fixture
first, then breaks exactly one thing — so a failure here names the check that
stopped working rather than a fixture that drifted.
"""

from __future__ import annotations

import dataclasses

import pytest

from conftest import make_component, make_op, make_vendor
from m3_production_delay.review.composer import compose
from m3_production_delay.review.evidence import build_evidence
from m3_production_delay.review.schemas import (
    SCOPE_JOB,
    SCOPE_OPERATION,
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_PREDECESSOR_OVERRUN,
    SIGNAL_SUPPLIER_RELIABILITY,
    SIGNAL_TIME_OVERRUN,
    InsightLine,
    has_errors,
)
from m3_production_delay.review.validators import (
    UNIT_MISMATCH_RATIO,
    check_cascade_context,
    check_forbidden_causes,
    check_no_omission,
    check_plausibility,
    check_quoted_numbers,
    check_signals_fired,
    check_summary_consistent,
    check_zero_fired,
    check_zero_weight_hidden,
    validate,
)


@pytest.fixture
def clean(section1_job, weights, threshold):
    pack = build_evidence(section1_job, weights, threshold)
    return pack, compose(pack)


def _checks(issues) -> set[str]:
    return {issue.check for issue in issues}


def _index_of(draft, signal_key: str) -> int:
    """Lines are ordered by contribution, so a test that wants to break the
    time line must find it rather than assume it is first."""
    return next(i for i, line in enumerate(draft.why_lines) if line.signal_key == signal_key)


def _replace_line(draft, signal_key, **changes):
    lines = list(draft.why_lines)
    index = _index_of(draft, signal_key)
    lines[index] = dataclasses.replace(lines[index], **changes)
    return draft.with_lines(tuple(lines))


# ─── the happy path ──────────────────────────────────────────────────────────


def test_the_template_path_is_validator_clean(clean):
    pack, draft = clean
    issues = validate(pack, draft)
    assert not has_errors(issues)
    # The fixture's not-started operation is scored delayed by the engine, so
    # exactly one plausibility warning and one cascade note are expected.
    assert _checks(issues) == {"not_started_but_delayed", "cascading_predecessor"}


# ─── signal exists and fired ─────────────────────────────────────────────────


def test_a_line_citing_an_unfired_signal_is_an_error(clean):
    pack, draft = clean
    # Welding's time signal exists but sits at 0.3 — well under its baseline.
    broken = _replace_line(draft, SIGNAL_TIME_OVERRUN, operation_id="wo-welding-0002")
    issues = check_signals_fired(pack, broken)
    assert [i.check for i in issues] == ["signal_not_fired"]
    assert issues[0].is_error


def test_a_line_citing_a_signal_with_no_evidence_is_an_error(clean):
    pack, draft = clean
    broken = _replace_line(draft, SIGNAL_TIME_OVERRUN, operation_id="wo-does-not-exist")
    issues = check_signals_fired(pack, broken)
    assert [i.check for i in issues] == ["signal_missing"]


# ─── omission ────────────────────────────────────────────────────────────────


def test_dropping_a_fired_weighted_line_is_an_omission_error(clean):
    pack, draft = clean
    kept = tuple(line for line in draft.why_lines if line.signal_key != SIGNAL_OPERATOR_PACE)
    issues = check_no_omission(pack, draft.with_lines(kept))
    assert [i.check for i in issues] == ["omitted_signal", "omitted_signal"]
    assert all(issue.is_error for issue in issues)


def test_a_fired_signal_that_cannot_be_rendered_is_a_warning_not_an_error(weights, threshold):
    # Supplier fired, but no component carries a vendor NAME, so no honest
    # sentence can be written: a warning, not a rejection.
    vendor = {"vendor_id": "v", "name": None, "vendor_lead_time_ratio": 2.0,
              "last_10_purchase_orders": []}
    op = make_op(
        actual_duration_minutes=600, time_overrun_ratio=1.25, operator_pace_ratio=None,
        material_shortfall_ratio=0.0, predecessor_time_overrun_ratio=None,
        composite_risk_score=1.3, is_delayed=True, predicted_overrun_hours=2.0,
        components=[make_component(100, 200, vendor)],
    )
    pack = build_evidence({"job_id": "WH/MO/00600", "operations": [op]}, weights, threshold)
    draft = compose(pack)
    issues = check_no_omission(pack, draft)
    assert [i.check for i in issues] == ["omitted_signal_unrenderable"]
    assert issues[0].severity == "warning"


def test_no_omission_is_reported_for_a_job_with_nothing_scorable(weights, threshold):
    op = make_op(
        actual_duration_minutes=None, time_overrun_ratio=None, operator_pace_ratio=1.5,
        material_shortfall_ratio=2.0, predecessor_time_overrun_ratio=None,
        composite_risk_score=1.6, is_delayed=True, predicted_overrun_hours=1.0,
        components=[make_component(100, 50)],
    )
    pack = build_evidence({"job_id": "WH/MO/00700", "operations": [op]}, weights, threshold)
    assert check_no_omission(pack, compose(pack)) == []


# ─── quoted numbers ──────────────────────────────────────────────────────────


def test_a_misquoted_number_is_an_error(clean):
    pack, draft = clean
    broken = _replace_line(
        draft, SIGNAL_TIME_OVERRUN, detail="Cutting · 14.0 hrs logged of 8.0 planned",
        quoted={"actual_hrs": 14.0, "expected_hrs": 8.0, "delta_hrs": 2.0},
    )
    issues = check_quoted_numbers(pack, broken)
    assert "quoted_number_mismatch" in _checks(issues)


def test_a_number_in_the_prose_that_is_quoted_nowhere_is_an_error(clean):
    pack, draft = clean
    broken = _replace_line(draft, SIGNAL_TIME_OVERRUN, delta="+2.0 hrs, 3 days late")
    issues = check_quoted_numbers(pack, broken)
    assert [i.check for i in issues] == ["unquoted_number"]


def test_quoting_a_field_the_signal_does_not_have_is_an_error(clean):
    pack, draft = clean
    broken = _replace_line(
        draft, SIGNAL_TIME_OVERRUN,
        quoted={"actual_hrs": 10.0, "expected_hrs": 8.0, "delta_hrs": 2.0,
                "machine_downtime_hrs": 3.0},
    )
    issues = check_quoted_numbers(pack, broken)
    assert [i.check for i in issues] == ["quoted_number_unknown"]


def test_rounding_within_the_display_step_is_not_a_misquote(weights, threshold):
    # 0.06 hrs renders as "0.1"; a pure 2% tolerance would call that a 67%
    # error, which is why the tolerance has an absolute floor.
    op = make_op(
        expected_duration_minutes=100, actual_duration_minutes=106, time_overrun_ratio=1.06,
        operator_pace_ratio=None, material_shortfall_ratio=0.0,
        predecessor_time_overrun_ratio=None, composite_risk_score=1.06, is_delayed=True,
        predicted_overrun_hours=0.1,
    )
    pack = build_evidence({"job_id": "WH/MO/00800", "operations": [op]}, weights, threshold)
    assert not has_errors(validate(pack, compose(pack)))


def test_a_part_number_is_not_mistaken_for_a_quoted_figure(clean):
    pack, draft = clean
    material = next(
        line for line in draft.why_lines if line.signal_key == SIGNAL_MATERIAL_SHORTFALL
    )
    # "Steel Plate 12mm" and "WH/MO/00142" both contain digits.
    assert "12mm" in material.detail and "00142" in material.detail
    assert check_quoted_numbers(pack, draft) == []


# ─── summary consistency ─────────────────────────────────────────────────────


def test_a_summary_that_disagrees_with_the_evidence_is_an_error(clean):
    pack, draft = clean
    broken = dataclasses.replace(draft, summary_risk_score=9.9)
    issues = check_summary_consistent(pack, broken)
    assert "summary_mismatch" in _checks(issues)


def test_a_badge_that_contradicts_the_threshold_is_an_error(clean):
    pack, draft = clean
    broken = dataclasses.replace(draft, summary_is_delayed=False)
    issues = check_summary_consistent(pack, broken)
    # Both: it no longer matches the evidence, and it no longer follows from
    # the score against the tenant's threshold.
    assert _checks(issues) == {"summary_mismatch", "threshold_inconsistent"}


# ─── zero-weight signals ─────────────────────────────────────────────────────


def test_showing_a_zero_weight_signal_as_a_cause_is_an_error(clean):
    pack, draft = clean
    cascade = InsightLine(
        index=len(draft.why_lines),
        signal_key=SIGNAL_PREDECESSOR_OVERRUN,
        scope=SCOPE_OPERATION,
        headline="A predecessor operation is running late",
        detail="Assembly · predecessor at 1.25× planned",
        operation_id="wo-assembly-0003",
        quoted={"ratio": 1.25},
    )
    broken = draft.with_lines(draft.why_lines + (cascade,))
    issues = check_zero_weight_hidden(pack, broken)
    assert [i.check for i in issues] == ["zero_weight_signal_shown"]


def test_a_tenant_weighting_a_signal_to_zero_hides_it(section1_job, threshold):
    weights = {"time_overrun_ratio": 1.0, "operator_pace_ratio": 0.0,
               "material_shortfall_ratio": 0.0, "supplier_reliability": 0.0}
    pack = build_evidence(section1_job, weights, threshold)
    draft = compose(pack)
    assert {line.signal_key for line in draft.why_lines} == {SIGNAL_TIME_OVERRUN}
    assert check_zero_weight_hidden(pack, draft) == []


# ─── forbidden causes ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phrase",
    ["machine breakdown", "quality hold", "rework", "approval", "inspection", "weather", "strike"],
)
def test_a_cause_this_system_cannot_observe_is_an_error(clean, phrase):
    pack, draft = clean
    broken = _replace_line(draft, SIGNAL_TIME_OVERRUN, headline=f"Delayed by {phrase} on the line")
    issues = check_forbidden_causes(pack, broken)
    assert [i.check for i in issues] == ["forbidden_cause"]


# ─── plausibility warnings ───────────────────────────────────────────────────


def test_a_possible_unit_mismatch_is_warned_about(weights, threshold):
    op = make_op(
        expected_duration_minutes=480,
        actual_duration_minutes=int(480 * UNIT_MISMATCH_RATIO) + 60,  # as if seconds
        time_overrun_ratio=21.1, operator_pace_ratio=None, material_shortfall_ratio=0.0,
        predecessor_time_overrun_ratio=None, composite_risk_score=21.1, is_delayed=True,
        predicted_overrun_hours=160.0,
    )
    pack = build_evidence({"job_id": "WH/MO/00900", "operations": [op]}, weights, threshold)
    issues = check_plausibility(pack, compose(pack))
    assert [i.check for i in issues] == ["implausible_duration"]
    assert "unit mismatch" in issues[0].message
    assert issues[0].severity == "warning"


def test_not_started_but_delayed_is_warned_about(clean):
    pack, draft = clean
    issues = check_plausibility(pack, draft)
    assert [i.check for i in issues] == ["not_started_but_delayed"]
    assert issues[0].ref == "wo-assembly-0003"


# ─── zero fired, but delayed ─────────────────────────────────────────────────


def test_delayed_with_nothing_fired_warns_and_names_the_top_signal(weights, threshold):
    # Operator pace of 1.15 is above 1.0 but below its 1.2 baseline, and it is
    # the only signal with a value, so the composite renormalises to 1.15 —
    # over the threshold with nothing to point at.
    op = make_op(
        actual_duration_minutes=None, time_overrun_ratio=None, operator_pace_ratio=1.15,
        material_shortfall_ratio=None, predecessor_time_overrun_ratio=None,
        composite_risk_score=1.15, is_delayed=True, predicted_overrun_hours=1.0,
    )
    scorable = make_op(
        operation_id="op-scorable", actual_duration_minutes=240,
        expected_duration_minutes=480, time_overrun_ratio=0.5, operator_pace_ratio=1.15,
        material_shortfall_ratio=None, predecessor_time_overrun_ratio=None,
        composite_risk_score=1.15, is_delayed=True, predicted_overrun_hours=1.0,
    )
    pack = build_evidence(
        {"job_id": "WH/MO/01000", "operations": [op, scorable]}, weights, threshold
    )
    draft = compose(pack)

    assert draft.summary_is_delayed is True
    assert draft.why_lines == ()
    issues = check_zero_fired(pack, draft)
    assert [i.check for i in issues] == ["no_fired_signal"]
    assert issues[0].severity == "warning"
    assert "operator_pace_ratio" in issues[0].message and "1.15" in issues[0].message


def test_no_zero_fired_warning_when_lines_exist(clean):
    pack, draft = clean
    assert draft.why_lines
    assert check_zero_fired(pack, draft) == []


# ─── cascading context ───────────────────────────────────────────────────────


def test_a_cascading_predecessor_is_reported_as_info(clean):
    pack, draft = clean
    issues = check_cascade_context(pack, draft)
    assert [i.check for i in issues] == ["cascading_predecessor"]
    assert issues[0].severity == "info"
    assert issues[0].ref == "wo-assembly-0003"
