"""Composer: the user story's line formats, filled only from evidence."""

from __future__ import annotations

import pytest

from conftest import make_component, make_op, make_vendor
from m3_production_delay.review.composer import HEADLINES, MAX_NAMED_COMPONENTS, compose
from m3_production_delay.review.evidence import build_evidence
from m3_production_delay.review.schemas import (
    SCOPE_JOB,
    SCOPE_OPERATION,
    SIGNAL_MATERIAL_SHORTFALL,
    SIGNAL_OPERATOR_PACE,
    SIGNAL_PREDECESSOR_OVERRUN,
    SIGNAL_SUPPLIER_RELIABILITY,
    SIGNAL_TIME_OVERRUN,
)


def _draft(job, weights, threshold):
    pack = build_evidence(job, weights, threshold)
    return pack, compose(pack)


def _line(draft, signal_key, operation_id=None):
    for line in draft.why_lines:
        if line.signal_key == signal_key and (
            operation_id is None or line.operation_id == operation_id
        ):
            return line
    return None


# ─── the four line formats ───────────────────────────────────────────────────


def test_time_line_matches_the_user_story_format(section1_job, weights, threshold):
    _, draft = _draft(section1_job, weights, threshold)
    line = _line(draft, SIGNAL_TIME_OVERRUN, "wo-cutting-0001")

    assert line.headline == "Actual time logged has exceeded expected duration"
    assert line.detail == "INDEPENDENT · wo-cutti · 10.0 hrs logged of 8.0 planned"
    assert line.delta == "+2.0 hrs"
    assert line.scope == SCOPE_OPERATION
    assert line.quoted == {"actual_hrs": 10.0, "expected_hrs": 8.0, "delta_hrs": 2.0}


def test_operator_line_matches_the_user_story_format(section1_job, weights, threshold):
    _, draft = _draft(section1_job, weights, threshold)
    line = _line(draft, SIGNAL_OPERATOR_PACE, "wo-cutting-0001")

    assert line.headline == "Assigned operator has a history of overrunning"
    assert line.detail.endswith("operator pace 1.25× planned over recent completed jobs")
    assert line.delta is None
    assert line.quoted == {"pace_ratio": 1.25}


def test_material_line_quotes_the_shortfall_quantity_not_the_ratio(
    section1_job, weights, threshold
):
    _, draft = _draft(section1_job, weights, threshold)
    line = _line(draft, SIGNAL_MATERIAL_SHORTFALL)

    assert line.headline == "Required material is short in the warehouse"
    assert line.detail == "WH/MO/00142 · Steel Plate 12mm – 60 short"
    assert line.scope == SCOPE_JOB
    assert line.quoted == {"shortfall:item-steel-plate": 60.0}
    # 2.5 is the ratio that went into the score; it is meaningless on screen
    # and must never be the number a planner reads.
    assert "2.5" not in line.detail


def test_supplier_line_names_the_vendors_the_mean_was_taken_over(
    section1_job, weights, threshold
):
    _, draft = _draft(section1_job, weights, threshold)
    line = _line(draft, SIGNAL_SUPPLIER_RELIABILITY)

    assert line.headline == "Supplier for a short component has a late-delivery history"
    # Both vendors are named because the engine's supplier mean is taken over
    # every component that has a lead-time ratio, not only the short ones —
    # 1.5 (late) and 0.79 (early) average to 1.14. Naming one and quoting the
    # other would be exactly the untraceable claim this design forbids.
    assert "Colombo Fasteners" in line.detail and "Lanka Steel" in line.detail
    assert line.detail.endswith("average lead-time ratio 1.14")
    assert line.quoted == {"mean_lead_time_ratio": pytest.approx(8.0 / 7.0, abs=1e-9)}


def test_no_recommendation_is_ever_composed(section1_job, weights, threshold):
    _, draft = _draft(section1_job, weights, threshold)
    text = " ".join(line.text for line in draft.why_lines).lower()
    for word in ("should", "recommend", "consider", "expedite", "reschedule"):
        assert word not in text


# ─── what earns a line ───────────────────────────────────────────────────────


def test_only_fired_signals_get_a_line(section1_job, weights, threshold):
    pack, draft = _draft(section1_job, weights, threshold)
    welding = pack.operation("wo-welding-0002")

    assert welding.is_scorable
    assert welding.signal(SIGNAL_TIME_OVERRUN).fired is False
    assert welding.signal(SIGNAL_OPERATOR_PACE).fired is False
    assert _line(draft, SIGNAL_TIME_OVERRUN, "wo-welding-0002") is None
    assert _line(draft, SIGNAL_OPERATOR_PACE, "wo-welding-0002") is None


def test_a_non_scorable_operation_gets_no_lines(section1_job, weights, threshold):
    pack, draft = _draft(section1_job, weights, threshold)
    assembly = pack.operation("wo-assembly-0003")

    assert assembly.is_scorable is False
    assert assembly.is_delayed is True  # the engine still scored it
    assert not [line for line in draft.why_lines if line.operation_id == "wo-assembly-0003"]


def test_a_job_with_no_scorable_operation_composes_no_lines_at_all(weights, threshold):
    op = make_op(
        actual_duration_minutes=None, time_overrun_ratio=None, operator_pace_ratio=1.5,
        material_shortfall_ratio=2.0, predecessor_time_overrun_ratio=None,
        composite_risk_score=1.6, is_delayed=True, predicted_overrun_hours=3.0,
        components=[make_component(100, 50)],
    )
    _, draft = _draft({"job_id": "WH/MO/00300", "operations": [op]}, weights, threshold)
    assert draft.why_lines == ()


def test_the_cascading_predecessor_never_becomes_a_line(section1_job, weights, threshold):
    pack, draft = _draft(section1_job, weights, threshold)
    assert pack.operation("wo-assembly-0003").signal(SIGNAL_PREDECESSOR_OVERRUN).fired is True
    assert all(line.signal_key != SIGNAL_PREDECESSOR_OVERRUN for line in draft.why_lines)


def test_a_zero_weight_signal_gets_no_line(section1_job, threshold):
    weights = {"time_overrun_ratio": 0.6, "operator_pace_ratio": 0.4,
               "material_shortfall_ratio": 0.0, "supplier_reliability": 0.0}
    _, draft = _draft(section1_job, weights, threshold)
    assert _line(draft, SIGNAL_MATERIAL_SHORTFALL) is None
    assert _line(draft, SIGNAL_SUPPLIER_RELIABILITY) is None


# ─── ordering, scope and the material list ───────────────────────────────────


def test_lines_are_ordered_by_contribution(section1_job, weights, threshold):
    _, draft = _draft(section1_job, weights, threshold)
    shares = [line.contribution for line in draft.why_lines]
    assert shares == sorted(shares, reverse=True)
    assert [line.index for line in draft.why_lines] == list(range(len(draft.why_lines)))


def test_job_scoped_lines_appear_once_however_many_operations_share_them(
    section1_job, weights, threshold
):
    _, draft = _draft(section1_job, weights, threshold)
    for key in (SIGNAL_MATERIAL_SHORTFALL, SIGNAL_SUPPLIER_RELIABILITY):
        assert sum(1 for line in draft.why_lines if line.signal_key == key) == 1


def test_material_overrun_is_shown_even_when_the_job_is_not_delayed(weights, threshold):
    vendor = make_vendor([])
    op = make_op(
        time_overrun_ratio=0.5, operator_pace_ratio=0.9, material_shortfall_ratio=2.0,
        predecessor_time_overrun_ratio=None, composite_risk_score=0.6, is_delayed=False,
        predicted_overrun_hours=-1.0, actual_duration_minutes=240,
        components=[make_component(100, 50, vendor)],
    )
    _, draft = _draft({"job_id": "WH/MO/00400", "operations": [op]}, weights, threshold)

    assert draft.summary_is_delayed is False
    # The shortage is a fact about the warehouse, not a consequence of the
    # badge — it is listed either way.
    assert [c.shortfall_quantity for c in draft.material_overrun] == [50.0]


def test_material_line_summarises_beyond_the_named_component_limit(weights, threshold):
    components = [
        {
            "component_id": f"item-{i}",
            "name": f"Part {i}",
            "required_quantity": 100.0,
            "available_quantity": 40.0,
            "vendor": None,
        }
        for i in range(MAX_NAMED_COMPONENTS + 2)
    ]
    op = make_op(
        time_overrun_ratio=1.5, operator_pace_ratio=None, material_shortfall_ratio=6.25,
        predecessor_time_overrun_ratio=None, composite_risk_score=1.4, is_delayed=True,
        predicted_overrun_hours=2.0, actual_duration_minutes=720, components=components,
    )
    _, draft = _draft({"job_id": "WH/MO/00500", "operations": [op]}, weights, threshold)
    line = _line(draft, SIGNAL_MATERIAL_SHORTFALL)

    assert line.detail.count("short") == MAX_NAMED_COMPONENTS
    assert line.detail.endswith("+2 more")
    # The summarised count is quoted too, so every numeral stays traceable.
    assert line.quoted["more_count"] == 2.0


def test_headline_table_covers_every_composable_signal():
    assert set(HEADLINES) == {
        SIGNAL_TIME_OVERRUN,
        SIGNAL_OPERATOR_PACE,
        SIGNAL_MATERIAL_SHORTFALL,
        SIGNAL_SUPPLIER_RELIABILITY,
    }


def test_composition_is_deterministic(section1_job, weights, threshold):
    _, first = _draft(section1_job, weights, threshold)
    _, second = _draft(section1_job, weights, threshold)
    assert first.to_dict() == second.to_dict()
