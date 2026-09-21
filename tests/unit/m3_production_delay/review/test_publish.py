"""Writeback payload: the shape that lands in
``manufacturing_orders.customElements``, checked without a database.

``publish_insights(..., dry_run=True)`` is the offline path — it builds and
logs exactly what would be written, so the payload contract is testable in CI
while the SQL itself stays covered by the needs_db integration tests.
"""

from __future__ import annotations

import json
import math

import pytest

from conftest import approving_agent, make_component, make_op, unparsable_agent
from m3_production_delay.review.pipeline import review_job, review_jobs
from m3_production_delay.review.publish import (
    ADVISORY_KEY,
    MODEL_VERSION,
    MODULE_DISPLAY_NAME,
    _audit_metadata,
    advisory_payload,
    publish_insights,
)
from m3_production_delay.review.schemas import (
    STATUS_FALLBACK_TEMPLATE,
    STATUS_SUPPRESSED_NOT_SCORABLE,
    ValidatedInsight,
)
from maxxflow_core.jsonutil import json_default


@pytest.fixture
def insight(section1_job, weights, threshold):
    return review_job(section1_job, weights, threshold, approving_agent())


# ─── payload shape ───────────────────────────────────────────────────────────


def test_the_payload_is_namespaced_under_one_key(insight):
    payload = advisory_payload(insight)
    assert list(payload) == [ADVISORY_KEY]
    assert ADVISORY_KEY == "ai_delay_insight"


def test_the_payload_carries_the_summary_the_panel_renders(insight):
    values = advisory_payload(insight)[ADVISORY_KEY]

    assert values["job_id"] == "WH/MO/00142"
    assert values["status"] == "approved_with_warnings"
    assert values["risk_score"] == pytest.approx(1.4523809523809523)
    assert values["overrun_hours"] == pytest.approx(8.666666666666666)
    assert values["is_delayed"] is True
    assert values["summary_basis"] == "worst_operation"
    assert values["delay_threshold"] == 1.0
    assert values["model_version"] == MODEL_VERSION == "m3-review-v1"
    assert values["scored_at"]  # from the platform clock, not now()


def test_every_line_in_the_payload_carries_its_traceable_numbers(insight):
    values = advisory_payload(insight)[ADVISORY_KEY]
    assert len(values["why_lines"]) == 4
    for line in values["why_lines"]:
        assert line["headline"] and line["detail"]
        assert line["quoted"], "a line must record the evidence values it quoted"
        assert line["scope"] in {"operation", "job"}


def test_material_overrun_rides_along_independently_of_the_lines(insight):
    values = advisory_payload(insight)[ADVISORY_KEY]
    assert [c["component_id"] for c in values["material_overrun"]] == ["item-steel-plate"]
    assert values["material_overrun"][0]["shortfall_quantity"] == 60.0


def test_the_payload_serialises_strictly(insight):
    encoded = json.dumps(advisory_payload(insight), default=json_default, allow_nan=False)
    assert "Infinity" not in encoded and "NaN" not in encoded
    assert json.loads(encoded)[ADVISORY_KEY]["job_id"] == "WH/MO/00142"


def test_a_zero_stock_job_still_serialises(weights, threshold):
    op = make_op(
        actual_duration_minutes=600, expected_duration_minutes=480, time_overrun_ratio=1.25,
        operator_pace_ratio=None, material_shortfall_ratio=math.inf,
        predecessor_time_overrun_ratio=None, composite_risk_score=math.inf, is_delayed=True,
        predicted_overrun_hours=2.0, components=[make_component(100, 0)],
    )
    insight = review_job(
        {"job_id": "WH/MO/02300", "operations": [op]}, weights, threshold, approving_agent()
    )
    encoded = json.dumps(advisory_payload(insight), default=json_default, allow_nan=False)

    assert "Infinity" not in encoded
    values = json.loads(encoded)[ADVISORY_KEY]
    assert values["risk_score"] is None
    # The shortage itself is not lost with the infinity.
    assert values["material_overrun"][0]["shortfall_quantity"] == 100.0
    assert any(i["check"] == "non_finite" for i in values["issues"])


def test_round_trips_through_json(insight):
    encoded = json.dumps(insight.to_dict(), default=json_default, allow_nan=False)
    restored = ValidatedInsight.from_dict(json.loads(encoded))
    assert restored.to_dict() == insight.to_dict()


# ─── auditing ────────────────────────────────────────────────────────────────


def test_only_the_three_unvalidated_statuses_are_audited(
    section1_job, weights, threshold
):
    approved = review_job(section1_job, weights, threshold, approving_agent())
    fallback = review_job(section1_job, weights, threshold, unparsable_agent())
    suppressed = review_job(
        {
            "job_id": "WH/MO/02400",
            "operations": [
                make_op(
                    actual_duration_minutes=None, time_overrun_ratio=None,
                    operator_pace_ratio=None, material_shortfall_ratio=0.0,
                    predecessor_time_overrun_ratio=None, composite_risk_score=None,
                    is_delayed=None, predicted_overrun_hours=None,
                )
            ],
        },
        weights, threshold, approving_agent(),
    )

    assert approved.needs_audit is False
    assert fallback.needs_audit is True and fallback.status == STATUS_FALLBACK_TEMPLATE
    assert suppressed.needs_audit is True
    assert suppressed.status == STATUS_SUPPRESSED_NOT_SCORABLE


def test_audit_metadata_explains_why_without_quoting_line_text(
    section1_job, weights, threshold
):
    insight = review_job(section1_job, weights, threshold, unparsable_agent())
    metadata = _audit_metadata(insight)

    assert metadata["job_reference"] == "WH/MO/00142"
    assert metadata["status"] == STATUS_FALLBACK_TEMPLATE
    assert metadata["judge_parse_error"].startswith("invalid_json")
    assert metadata["delay_threshold"] == 1.0
    assert any(i["check"] == "judge_rejected" for i in metadata["issues"])
    # Info-level context is not an audit reason.
    assert all(i["severity"] != "info" for i in metadata["issues"])
    assert json.dumps(metadata, default=json_default, allow_nan=False)
    assert MODULE_DISPLAY_NAME == "Production Delay Insights"


# ─── dry run ─────────────────────────────────────────────────────────────────


def test_dry_run_touches_no_database(section1_job, weights, threshold):
    insights = review_jobs([section1_job], weights, threshold, approving_agent())
    # No DATA_DB_URL, no Postgres: a dry run that reached get_data_access()
    # would fail here rather than return.
    assert publish_insights(insights, tenant="demo", dry_run=True) == 1


def test_publishing_nothing_is_a_no_op():
    assert publish_insights([], tenant="demo") == 0
