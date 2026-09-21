"""M3 Section 3 — Validated AI Insight (the deterministic half).

Evidence from a scored job, template explanation lines, deterministic
validators, and the writeback. The judging call itself is not here: it belongs
to :mod:`m3_production_delay.llm_agents.review_agent`, and is composed with
this package by :class:`~m3_production_delay.orchestrator
.ProductionDelayOrchestrator`, the same way the Weight Agent is.

That split is the design, not an accident of layout. Nothing in this package
calls an LLM, and the agent it hands drafts to can only answer yes/no about
lines this package already wrote — so "no number is ever produced by a
language model" is a property of the dependency direction rather than of a
prompt instruction. If the judge cannot be reached, cannot be parsed, or keeps
refusing, these deterministic templates ship as ``fallback_template``.

See ``review/README.md`` for the contract and how to run one job end to end.
"""

from m3_production_delay.review.composer import compose
from m3_production_delay.review.evidence import build_evidence
from m3_production_delay.review.pipeline import review_job, review_jobs, run
from m3_production_delay.review.publish import advisory_payload, publish_insights
from m3_production_delay.review.schemas import (
    EvidencePack,
    InsightDraft,
    InsightLine,
    Issue,
    JudgeVerdict,
    ValidatedInsight,
)
from m3_production_delay.review.validators import validate

__all__ = [
    "build_evidence",
    "compose",
    "validate",
    "review_job",
    "review_jobs",
    "run",
    "advisory_payload",
    "publish_insights",
    "EvidencePack",
    "InsightDraft",
    "InsightLine",
    "Issue",
    "JudgeVerdict",
    "ValidatedInsight",
]
