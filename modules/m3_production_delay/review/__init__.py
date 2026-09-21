"""M3 Section 3 — Review Agent & Validated AI Insight.

The LLM-as-a-judge layer over the Risk Engine's output: deterministic evidence
from a scored job, template explanation lines, deterministic validators, and
then one LLM call that is only ever asked a yes/no question about whether each
line is supported by that evidence.

No number, signal or line of user-facing text is ever produced by the LLM — it
can approve lines, or cause them to be dropped, and nothing else. If the judge
cannot be reached, cannot be parsed, or keeps refusing, the deterministic
templates ship as ``fallback_template``: the panel degrades to the explanation
that already passed every deterministic check, never to silence.

See ``review/README.md`` for the input/output contract and how to run one job
end to end.
"""

from m3_production_delay.review.composer import compose
from m3_production_delay.review.evidence import build_evidence
from m3_production_delay.review.judge import build_llm, judge_draft
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
    "build_llm",
    "judge_draft",
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
