"""Pipeline: how each judge outcome maps to a status, and the two paths that
never reach the judge at all.

Every judge here is a real ReviewAgent driven by a scripted provider, so the
agent's own prompt assembly and verdict parsing run in each case. That is the
point: a mock would assert that the agent was called, and what matters is what
the pipeline does with what comes back.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from conftest import (
    agent_for,
    approving_agent,
    exploding_agent,
    make_component,
    make_op,
    never_called_agent,
    rejecting_agent,
    rejecting_once_agent,
    responding_agent,
    unparsable_agent,
)
from m3_production_delay.review.composer import compose
from m3_production_delay.review.evidence import build_evidence
from m3_production_delay.llm_agents.review_agent import ReviewAgent, build_config
from m3_production_delay.review.pipeline import review_job, review_jobs
from m3_production_delay.review.schemas import (
    STATUS_APPROVED,
    STATUS_APPROVED_WITH_WARNINGS,
    STATUS_FALLBACK_TEMPLATE,
    STATUS_REJECTED,
    STATUS_SUPPRESSED_NOT_SCORABLE,
    SIGNAL_TIME_OVERRUN,
)


def _checks(insight) -> set[str]:
    return {issue.check for issue in insight.issues}


# ─── judge outcomes ──────────────────────────────────────────────────────────


def test_an_approved_insight_keeps_every_template_line(section1_job, weights, threshold):
    insight = review_job(section1_job, weights, threshold, approving_agent())

    # The fixture's not-started operation carries a plausibility warning, so
    # the clean status here is "approved_with_warnings" rather than "approved".
    assert insight.status == STATUS_APPROVED_WITH_WARNINGS
    assert insight.attempts == 1
    assert len(insight.why_lines) == 5
    assert insight.judge.approved is True
    assert insight.judge.skipped is False


def test_a_warning_free_job_is_plainly_approved(weights, threshold):
    op = make_op(
        actual_duration_minutes=600, expected_duration_minutes=480, time_overrun_ratio=1.25,
        operator_pace_ratio=None, material_shortfall_ratio=0.0,
        predecessor_time_overrun_ratio=None, composite_risk_score=1.25, is_delayed=True,
        predicted_overrun_hours=2.0,
    )
    insight = review_job(
        {"job_id": "WH/MO/02000", "operations": [op]}, weights, threshold, approving_agent()
    )
    assert insight.status == STATUS_APPROVED
    assert insight.issues == ()


def test_an_unsupported_line_is_dropped_and_the_rest_re_judged(
    section1_job, weights, threshold
):
    insight = review_job(section1_job, weights, threshold, rejecting_once_agent(0))

    assert insight.status == STATUS_APPROVED_WITH_WARNINGS
    assert insight.attempts == 2
    assert len(insight.why_lines) == 4  # one dropped
    assert "judge_dropped_lines" in _checks(insight)
    assert [line.index for line in insight.why_lines] == [0, 1, 2, 3]


def test_a_judge_that_keeps_refusing_falls_back_to_the_full_template(
    section1_job, weights, threshold
):
    insight = review_job(section1_job, weights, threshold, rejecting_agent(0))

    assert insight.status == STATUS_FALLBACK_TEMPLATE
    assert insight.attempts == 2
    # The FULL template set, not the pruned one: those lines already passed
    # every deterministic check, and an unconvinced judge is no reason to
    # show the operator less.
    assert len(insight.why_lines) == 5
    assert "judge_rejected" in _checks(insight)


def test_an_unparsable_verdict_is_never_read_as_approval(section1_job, weights, threshold):
    insight = review_job(section1_job, weights, threshold, unparsable_agent())

    assert insight.status == STATUS_FALLBACK_TEMPLATE
    assert insight.judge.approved is False
    assert insight.judge.parse_error.startswith("invalid_json")
    assert insight.attempts == 1  # nothing to drop, so no retry


def test_a_provider_failure_degrades_to_the_template(section1_job, weights, threshold):
    insight = review_job(section1_job, weights, threshold, exploding_agent())

    assert insight.status == STATUS_FALLBACK_TEMPLATE
    assert insight.judge.parse_error.startswith("provider_error")
    assert len(insight.why_lines) == 5


def test_the_default_stub_provider_always_produces_a_fallback(
    section1_job, weights, threshold
):
    # LLM_PROVIDER=stub returns prose, not JSON. Documented behaviour, and the
    # reason CI never sees an "approved" status without a scripted judge.
    insight = review_job(section1_job, weights, threshold, ReviewAgent())
    assert insight.status == STATUS_FALLBACK_TEMPLATE


# ─── paths that never reach the judge ────────────────────────────────────────


def test_a_job_with_nothing_scorable_is_suppressed(weights, threshold):
    op = make_op(
        actual_duration_minutes=None, time_overrun_ratio=None, operator_pace_ratio=1.5,
        current_done_quantity=0,
        material_shortfall_ratio=2.0, predecessor_time_overrun_ratio=None,
        composite_risk_score=1.6, is_delayed=True, predicted_overrun_hours=3.0,
        status="NOT_STARTED", components=[make_component(100, 40)],
    )

    insight = review_job(
        {"job_id": "WH/MO/02100", "operations": [op]}, weights, threshold, never_called_agent()
    )

    assert insight.status == STATUS_SUPPRESSED_NOT_SCORABLE
    assert insight.why_lines == ()
    assert insight.attempts == 0
    assert insight.needs_audit is True
    # The engine's own numbers still ride along; this agent simply does not
    # vouch for an explanation of them.
    assert insight.material_overrun[0].shortfall_quantity == 60.0
    assert "not_started_but_delayed" in _checks(insight)


def test_a_validator_error_rejects_without_calling_the_judge(
    section1_job, weights, threshold, monkeypatch
):
    def tampering_compose(pack):
        """Stands in for a composer bug: a line whose number is not the
        evidence's. Simulated here because the real composer cannot produce
        one — which is exactly why this path is not retryable."""
        draft = compose(pack)
        lines = list(draft.why_lines)
        index = next(
            i for i, line in enumerate(lines) if line.signal_key == SIGNAL_TIME_OVERRUN
        )
        lines[index] = dataclasses.replace(
            lines[index], quoted={**lines[index].quoted, "actual_hrs": 99.0}
        )
        return draft.with_lines(tuple(lines))

    monkeypatch.setattr("m3_production_delay.review.pipeline.compose", tampering_compose)

    insight = review_job(section1_job, weights, threshold, never_called_agent())

    assert insight.status == STATUS_REJECTED
    assert insight.why_lines == ()
    assert insight.attempts == 0
    assert insight.needs_audit is True
    assert "quoted_number_mismatch" in _checks(insight)


def test_delayed_with_no_fired_signal_ships_the_summary_alone(weights, threshold):
    op = make_op(
        actual_duration_minutes=240, expected_duration_minutes=480, time_overrun_ratio=0.5,
        operator_pace_ratio=1.15, material_shortfall_ratio=None,
        predecessor_time_overrun_ratio=None, composite_risk_score=1.15, is_delayed=True,
        predicted_overrun_hours=1.0,
    )
    insight = review_job(
        {"job_id": "WH/MO/02200", "operations": [op]}, weights, threshold, approving_agent()
    )

    assert insight.status == STATUS_APPROVED_WITH_WARNINGS
    assert insight.why_lines == ()
    assert insight.is_delayed is True
    assert insight.judge.skipped is True  # nothing to judge
    assert "no_fired_signal" in _checks(insight)


# ─── verdict parsing ─────────────────────────────────────────────────────────


@pytest.fixture
def drafted(section1_job, weights, threshold):
    pack = build_evidence(section1_job, weights, threshold)
    return pack, compose(pack)


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        ("not json at all", "invalid_json"),
        ("[1, 2, 3]", "schema_mismatch"),
        ('{"lines": []}', "schema_mismatch"),
        ('{"approved": "yes", "lines": []}', "schema_mismatch"),
        ('{"approved": true, "lines": {}}', "schema_mismatch"),
        ('{"approved": true, "lines": [{"line_index": 99, "supported": true}]}', "out_of_range"),
        ('{"approved": true, "lines": [{"line_index": 0, "supported": "yes"}]}', "schema_mismatch"),
        ('{"approved": true, "lines": [{"line_index": 0, "supported": true}]}', "incomplete_verdict"),
    ],
)
def test_every_malformed_verdict_is_unapproved(drafted, response, expected_error):
    pack, draft = drafted
    verdict = responding_agent(response).judge(pack, draft)
    assert verdict.approved is False
    assert expected_error in verdict.parse_error


def test_a_duplicate_line_index_is_rejected(drafted):
    pack, draft = drafted
    response = json.dumps(
        {
            "approved": True,
            "lines": [{"line_index": 0, "supported": True}] * 2,
        }
    )
    verdict = responding_agent(response).judge(pack, draft)
    assert "duplicate_line_index" in verdict.parse_error


def test_an_approval_that_contradicts_itself_is_read_conservatively(drafted):
    pack, draft = drafted
    response = json.dumps(
        {
            "approved": True,
            "lines": [
                {"line_index": i, "supported": i != 1, "evidence_ref": None, "issue": None}
                for i in range(len(draft.why_lines))
            ],
            "unsupported_claims": [],
            "omitted_signals": [],
        }
    )
    verdict = responding_agent(response).judge(pack, draft)

    assert verdict.approved is False  # the dissent wins, not the blanket yes
    assert verdict.unsupported_indices == (1,)


def test_a_claimed_omission_blocks_approval(drafted):
    pack, draft = drafted
    response = json.dumps(
        {
            "approved": True,
            "lines": [
                {"line_index": i, "supported": True} for i in range(len(draft.why_lines))
            ],
            "unsupported_claims": [],
            "omitted_signals": ["material_shortfall_ratio"],
        }
    )
    verdict = responding_agent(response).judge(pack, draft)
    assert verdict.approved is False
    assert verdict.omitted_signals == ("material_shortfall_ratio",)


def test_the_prompt_carries_the_evidence_and_the_indexed_lines(drafted):
    pack, draft = drafted
    agent = approving_agent()
    agent.judge(pack, draft)

    # The platform's LLMProvider port takes a single prompt, so the agent
    # concatenates its two halves; the provider records what it was actually
    # sent.
    prompt = agent._llm_provider.calls[0]
    assert "EVIDENCE PACK" in prompt
    assert pack.job_id in prompt
    assert all(f"[{line.index}] signal=" in prompt for line in draft.why_lines)
    assert "fired" in prompt.lower()
    # The judge is told what NOT to check, so a missing recommendation can
    # never be read as a defect.
    assert "recommendations" in prompt.lower()


def test_an_empty_draft_is_not_sent_to_the_judge(drafted):
    pack, draft = drafted

    verdict = never_called_agent().judge(pack, draft.with_lines(()))
    assert verdict.approved is True
    assert verdict.skipped is True


# ─── batch ───────────────────────────────────────────────────────────────────


def test_review_jobs_reviews_every_job_independently(section1_job, weights, threshold):
    other = json.loads(json.dumps(section1_job))
    other["job_id"] = "WH/MO/00143"
    insights = review_jobs([section1_job, other], weights, threshold, approving_agent())

    assert [i.job_id for i in insights] == ["WH/MO/00142", "WH/MO/00143"]
    assert {i.status for i in insights} == {STATUS_APPROVED_WITH_WARNINGS}


def test_one_failing_job_does_not_affect_the_next(section1_job, weights, threshold):
    unscorable = {
        "job_id": "WH/MO/00144",
        "operations": [
            make_op(
                actual_duration_minutes=None, time_overrun_ratio=None,
                current_done_quantity=0,
                operator_pace_ratio=None, material_shortfall_ratio=0.0,
                predecessor_time_overrun_ratio=None, composite_risk_score=None,
                is_delayed=None, predicted_overrun_hours=None,
            )
        ],
    }
    insights = review_jobs([unscorable, section1_job], weights, threshold, approving_agent())

    assert insights[0].status == STATUS_SUPPRESSED_NOT_SCORABLE
    assert insights[1].status == STATUS_APPROVED_WITH_WARNINGS
    assert len(insights[1].why_lines) == 5


def test_the_retry_names_the_dropped_signal_to_the_judge(section1_job, weights, threshold):
    """The pipeline must tell the second pass which signals it withdrew, or the
    re-judge is unanswerable — see test_review_agent for why."""
    seen: list[tuple[str, ...]] = []

    class Recording(ReviewAgent):
        def judge(self, pack, draft, *, withdrawn=(), tracer=None):
            seen.append(withdrawn)
            return super().judge(pack, draft, withdrawn=withdrawn)

    agent = rejecting_once_agent(0)
    recorder = Recording(llm_provider=agent._llm_provider, config=agent.config)
    insight = review_job(section1_job, weights, threshold, recorder)

    assert len(seen) == 2, "one judging call, then one re-judge"
    assert seen[0] == (), "the first pass has nothing to declare"
    assert seen[1], "the re-judge is told which signal's line was withdrawn"
    assert insight.status == STATUS_APPROVED_WITH_WARNINGS
