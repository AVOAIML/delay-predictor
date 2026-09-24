"""The orchestrator as the place both M3 agents are composed.

The Weight Agent side is covered by ``test_orchestrator.py``; this file covers
the Review Agent side and the one property that only shows up when both are
built by the same object: a single injected provider drives every LLM call M3
makes, which is what lets a caller run the whole module offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3_production_delay.llm_agents.review_agent import ReviewAgent, build_config
from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER
from m3_production_delay.orchestrator import ProductionDelayOrchestrator, WeightAgentRequest
from m3_production_delay.review.schemas import STATUS_APPROVED_WITH_WARNINGS, ValidatedInsight

_REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    _REPO_ROOT / "modules" / "m3_production_delay" / "review" / "fixtures" / "section1_job.json"
)
WEIGHTS = {
    "time_overrun_ratio": 0.40,
    "operator_pace_ratio": 0.30,
    "material_shortfall_ratio": 0.15,
    "supplier_reliability": 0.05,
}
THRESHOLD = 1.0
ALL_AVAILABLE = {signal: True for signal in SIGNAL_ORDER}


class ApprovingProvider:
    """Scripted provider answering the judge's one question, in the same
    style as ``orchestrator._DemoScriptedLLMProvider``."""

    name = "scripted (test only)"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, *, max_tokens: int = 256, generation_config=None) -> str:
        self.calls.append(prompt)
        count = prompt.count("] signal=")
        return json.dumps(
            {
                "approved": True,
                "lines": [{"line_index": i, "supported": True} for i in range(count)],
                "unsupported_claims": [],
                "omitted_signals": [],
            }
        )


@pytest.fixture
def scored_job() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _orchestrator(provider=None) -> ProductionDelayOrchestrator:
    return ProductionDelayOrchestrator(
        llm_provider=provider,
        review_config=build_config(llm_enabled=True),
    )


# ─── composition ─────────────────────────────────────────────────────────────


def test_the_orchestrator_builds_a_review_agent():
    assert isinstance(_orchestrator().review_agent, ReviewAgent)


def test_one_injected_provider_drives_both_agents():
    provider = ApprovingProvider()
    orchestrator = _orchestrator(provider)
    # The Review Agent resolves lazily, so reaching into it is how we confirm
    # the same provider was handed to both rather than each resolving its own.
    assert orchestrator.review_agent._llm_provider is provider


def test_a_prebuilt_agent_wins_over_the_shared_provider():
    """A caller that needs the two agents on different providers or configs
    can hand in a finished one."""
    custom = ReviewAgent(llm_provider=ApprovingProvider(), config=build_config(max_tokens=123))
    orchestrator = ProductionDelayOrchestrator(
        llm_provider=ApprovingProvider(), review_agent=custom
    )
    assert orchestrator.review_agent is custom
    assert orchestrator.review_agent.config.max_tokens == 123


# ─── review_jobs ─────────────────────────────────────────────────────────────


def test_review_jobs_returns_one_validated_insight_per_job(scored_job):
    orchestrator = _orchestrator(ApprovingProvider())
    insights = orchestrator.review_jobs(
        [scored_job], weights=WEIGHTS, threshold=THRESHOLD
    )

    assert len(insights) == 1
    assert isinstance(insights[0], ValidatedInsight)
    assert insights[0].job_id == "WH/MO/00142"
    assert insights[0].status == STATUS_APPROVED_WITH_WARNINGS
    assert len(insights[0].why_lines) == 5


def test_the_threshold_is_carried_into_the_insight(scored_job):
    orchestrator = _orchestrator(ApprovingProvider())
    insights = orchestrator.review_jobs([scored_job], weights=WEIGHTS, threshold=1.3)

    # 1.61 is above 1.3 as it was above 1.0, so the engine's badge still
    # holds — and the cutoff it was judged against travels with the payload.
    assert insights[0].delay_threshold == 1.3
    assert insights[0].is_delayed is True


def test_a_threshold_that_did_not_produce_the_scores_is_refused(scored_job):
    """`is_delayed` is the engine's verdict at the cutoff it scored with.
    Reviewing the same job at 2.0 would publish a badge attributed to a number
    that never produced it, so the mismatch is refused at the boundary rather
    than rationalised downstream."""
    orchestrator = _orchestrator(ApprovingProvider())
    with pytest.raises(ValueError, match="same delay_threshold"):
        orchestrator.review_jobs([scored_job], weights=WEIGHTS, threshold=2.0)


def test_reviewing_nothing_is_not_an_error():
    assert _orchestrator(ApprovingProvider()).review_jobs(
        [], weights=WEIGHTS, threshold=THRESHOLD
    ) == []


def test_the_orchestrator_does_no_io_of_its_own(scored_job):
    """No database is reachable in this test run, so a review that touched
    one would fail here rather than return — the reading and the writeback
    belong to review/pipeline.py."""
    insights = _orchestrator(ApprovingProvider()).review_jobs(
        [scored_job], weights=WEIGHTS, threshold=THRESHOLD
    )
    assert insights[0].why_lines


def test_weights_and_review_compose_through_one_orchestrator(scored_job):
    """The end-to-end shape `review/pipeline.run()` uses: resolve the tenant's
    weights, then review the scores those weights produced."""
    orchestrator = _orchestrator(ApprovingProvider())
    weights = orchestrator.resolve_risk_weights(
        WeightAgentRequest(
            tenant_id="tenant-review",
            availability=ALL_AVAILABLE,
            configured_bp={
                "time_overrun": 4000,
                "operator_skill": 3000,
                "seasonality": 1000,
                "material_availability": 1500,
                "supplier_reliability": 500,
            },
        )
    )
    assert weights == WEIGHTS

    insights = orchestrator.review_jobs(
        [scored_job], weights=weights, threshold=THRESHOLD
    )
    # The explanation is attributed against the same vector that was resolved,
    # never a default one.
    assert insights[0].status == STATUS_APPROVED_WITH_WARNINGS
    assert sum(line.contribution for line in insights[0].why_lines) > 0
