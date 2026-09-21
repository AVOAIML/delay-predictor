"""Section 3's orchestration: one scored job in, one
:class:`~m3_production_delay.review.schemas.ValidatedInsight` out, plus the
batch entry point that runs the whole chain for a tenant.

    evidence -> compose -> validate -> judge -> (drop unsupported, re-judge) -> publish

Two of those arrows never retry, on purpose:

  * **A validator error is not retryable.** It means the composer wrote a line
    the evidence does not support, or the evidence itself is inconsistent —
    a bug in this package, not a wording the judge could rescue. The job is
    ``rejected``, carries no lines, and lands in ``audit_logs`` where it can
    be found.
  * **A judge rejection retries exactly once**, and only by *removing* the
    lines the judge would not support. Nothing is rewritten between attempts,
    because nothing in this system except the composer may write a line. If
    the second verdict still refuses, the deterministic templates ship as
    ``fallback_template`` — the panel degrades to the explanation that already
    passed every deterministic check, never to silence.

``threshold`` has no default at any level of this module. The same composite
score means "delayed" or "fine" depending on the tenant, so it is always
supplied explicitly, recorded in the evidence pack, and published alongside
the verdict.
"""

from __future__ import annotations

import argparse
import json

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger
from maxxflow_core.jsonutil import json_default

from m3_production_delay.llm_agents.review_agent import ReviewAgent
from m3_production_delay.review.composer import compose
from m3_production_delay.review.evidence import build_evidence
from m3_production_delay.review.publish import MODEL_VERSION, advisory_payload, publish_insights
from m3_production_delay.review.schemas import (
    SEVERITY_WARNING,
    STATUS_APPROVED,
    STATUS_APPROVED_WITH_WARNINGS,
    STATUS_FALLBACK_TEMPLATE,
    STATUS_REJECTED,
    STATUS_SUPPRESSED_NOT_SCORABLE,
    EvidencePack,
    InsightDraft,
    Issue,
    JudgeVerdict,
    ValidatedInsight,
    has_errors,
)
from m3_production_delay.review.validators import validate

log = get_logger("m3_production_delay.review.pipeline")


def _insight(
    pack: EvidencePack,
    draft: InsightDraft | None,
    status: str,
    issues: tuple[Issue, ...],
    verdict: JudgeVerdict,
    attempts: int,
) -> ValidatedInsight:
    keeps_lines = status not in (STATUS_REJECTED, STATUS_SUPPRESSED_NOT_SCORABLE)
    return ValidatedInsight(
        job_id=pack.job_id,
        status=status,
        overrun_hours=pack.summary_overrun_hours,
        risk_score=pack.summary_risk_score,
        is_delayed=pack.summary_is_delayed,
        summary_basis=pack.summary_basis,
        delay_threshold=pack.delay_threshold,
        why_lines=draft.why_lines if (draft is not None and keeps_lines) else (),
        # Independent of the score and of the verdict: a short component is a
        # fact about the warehouse, worth showing on a job nobody judged.
        material_overrun=pack.material_overrun,
        issues=issues,
        judge=verdict,
        attempts=attempts,
        model_version=MODEL_VERSION,
        generated_at=get_clock().as_of().isoformat(),
    )


def review_job(
    job: dict, weights: dict[str, float], threshold: float, agent: ReviewAgent
) -> ValidatedInsight:
    """Review one scored job — one element of
    ``calculate_delay_elements_for_jobs()``'s output.

    ``agent`` is a :class:`~m3_production_delay.llm_agents.review_agent
    .ReviewAgent`; the orchestrator builds one, and tests inject a scripted
    ``LLMProvider`` into it the way ``test_m3_full_pipeline`` does for the
    Weight Agent. The attempt budget comes from the agent's own config, so
    the number of calls this loop can spend is stated in exactly one place.
    """
    pack = build_evidence(job, weights, threshold)
    log.info(
        "m3_review_evidence job_id=%s risk_score=%s overrun_hours=%s "
        "is_delayed=%s threshold=%s scorable_operations=%d",
        pack.job_id,
        pack.summary_risk_score,
        pack.summary_overrun_hours,
        pack.summary_is_delayed,
        pack.delay_threshold,
        len(pack.scorable_operations),
    )

    if not pack.scorable_operations:
        # Nothing in this job has enough logged progress to attribute a cause
        # to. The engine's own numbers still ride along in the payload; this
        # agent simply does not vouch for an explanation of them.
        issues = pack.issues + tuple(validate(pack, compose(pack)))
        return _insight(
            pack,
            None,
            STATUS_SUPPRESSED_NOT_SCORABLE,
            issues,
            JudgeVerdict(approved=False, skipped=True),
            attempts=0,
        )

    draft = compose(pack)
    log.info(
        "m3_review_composition job_id=%s candidate_lines=%d signals=%s",
        pack.job_id,
        len(draft.why_lines),
        [line.signal_key for line in draft.why_lines],
    )
    issues = pack.issues + tuple(validate(pack, draft))
    if has_errors(issues):
        return _insight(
            pack, draft, STATUS_REJECTED, issues, JudgeVerdict(approved=False, skipped=True), 0
        )

    judged = draft
    verdict = JudgeVerdict(approved=False, skipped=True)
    attempts = 0
    max_attempts = agent.config.max_attempts
    # Signals whose lines this judge asked to have removed. Passed back to it
    # on the re-judge, because every line the composer emits is for a fired,
    # weighted signal — so dropping one would otherwise trip the judge's own
    # "nothing omitted" check and make the second verdict fail by construction.
    withdrawn: tuple[str, ...] = ()
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        verdict = agent.judge(pack, judged, withdrawn=withdrawn)
        log.info(
            "m3_review_judge job_id=%s attempt=%d candidate_lines=%d approved=%s "
            "skipped=%s unsupported_indices=%s omitted_signals=%s parse_error=%s",
            pack.job_id,
            attempt,
            len(judged.why_lines),
            verdict.approved,
            verdict.skipped,
            verdict.unsupported_indices,
            verdict.omitted_signals,
            verdict.parse_error,
        )
        if verdict.approved:
            break
        unsupported = set(verdict.unsupported_indices)
        if not unsupported or attempt == max_attempts:
            break
        withdrawn += tuple(
            line.signal_key for line in judged.why_lines if line.index in unsupported
        )
        kept = tuple(line for line in judged.why_lines if line.index not in unsupported)
        issues += (
            Issue(
                check="judge_dropped_lines",
                severity=SEVERITY_WARNING,
                message=(
                    f"the judge did not support {len(unsupported)} of {len(judged.why_lines)} "
                    "lines; they were dropped and the remainder re-judged"
                ),
            ),
        )
        judged = judged.with_lines(kept)
        if not judged.why_lines:
            # Nothing left to judge. Re-asking about an empty list cannot
            # change the outcome, so fall back to the full template set.
            break

    if verdict.approved:
        dropped = len(draft.why_lines) - len(judged.why_lines)
        status = (
            STATUS_APPROVED
            if not issues and dropped == 0
            else STATUS_APPROVED_WITH_WARNINGS
        )
        return _insight(pack, judged, status, issues, verdict, attempts)

    issues += (
        Issue(
            check="judge_rejected",
            severity=SEVERITY_WARNING,
            message=(
                "the judge did not approve the insight after "
                f"{attempts} attempt(s)"
                + (f" ({verdict.parse_error})" if verdict.parse_error else "")
                + "; the deterministic template lines were published unjudged"
            ),
        ),
    )
    # Fall back to the FULL template draft, not the pruned one: those lines
    # already passed every deterministic check, and an unreadable or
    # unreachable judge is no reason to show the operator less.
    return _insight(pack, draft, STATUS_FALLBACK_TEMPLATE, issues, verdict, attempts)


def review_jobs(
    jobs: list[dict], weights: dict[str, float], threshold: float, agent: ReviewAgent
) -> list[ValidatedInsight]:
    """Review every scored job in a batch. One job's evidence, verdict and
    failures never affect another's — the batch is a loop, not a shared
    context."""
    insights = [review_job(job, weights, threshold, agent) for job in jobs]
    by_status: dict[str, int] = {}
    for insight in insights:
        by_status[insight.status] = by_status.get(insight.status, 0) + 1
    log.info("m3_review reviewed jobs=%d statuses=%s", len(insights), by_status)
    return insights


# ---------------------------------------------------------------------------
# Batch entry point: read -> rollup+elements (Section 1) -> weights (Section 2)
# -> review (Section 3) -> writeback. Mirrors m2_inventory.batch_scoring.score.
# ---------------------------------------------------------------------------


def run(
    *,
    tenant: str,
    threshold: float,
    job_references: list[str] | None = None,
    dry_run: bool = False,
) -> list[ValidatedInsight]:
    """Score, review and publish every (or some) job for one tenant.

    This function owns the IO — the tenant read, the rollup, the writeback —
    and nothing else. Both agents are reached through
    :class:`~m3_production_delay.orchestrator.ProductionDelayOrchestrator`,
    which is where agent composition lives: weights from the Weight Agent,
    then the review from the Review Agent, over the scores that same weight
    vector produced.

    A tenant with no computable signal at all raises
    ``AllSignalsUnavailableError`` from the weight call, unchanged — there is
    no weight vector to substitute for it, so there is nothing to explain
    either.
    """
    from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER
    from m3_production_delay.orchestrator import ProductionDelayOrchestrator, WeightAgentRequest
    from m3_production_delay.rule_engine.dal import read_delay_tables
    from m3_production_delay.rule_engine.elements import calculate_delay_elements_for_jobs
    from m3_production_delay.rule_engine.rollup import build_job_rollups

    tables, md = read_delay_tables(tenant)
    rollups = build_job_rollups(tables, md, job_references=job_references)
    if not rollups:
        log.info("m3_review no jobs to review tenant=%s", tenant)
        return []

    orchestrator = ProductionDelayOrchestrator()
    weights = orchestrator.resolve_risk_weights(
        WeightAgentRequest(
            tenant_id=tenant, availability={signal: True for signal in SIGNAL_ORDER}
        )
    )
    scored = calculate_delay_elements_for_jobs(
        rollups, risk_weights=weights, delay_threshold=threshold
    )
    insights = orchestrator.review_jobs(scored, weights=weights, threshold=threshold)
    publish_insights(insights, tenant=tenant, dry_run=dry_run)
    return insights


def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m m3_production_delay.review",
        description=(
            "Review the Risk Engine's delay scores for one tenant and write the validated "
            "insight to manufacturing_orders.customElements."
        ),
    )
    parser.add_argument("--tenant", default="demo")
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help=(
            "composite risk score above which a job counts as delayed. Required: there is no "
            "calibrated default, and the same score means different things per tenant."
        ),
    )
    parser.add_argument(
        "--job",
        action="append",
        dest="job_references",
        help="job reference to review (repeatable); omit to review every job",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the writeback payloads instead of writing them",
    )
    args = parser.parse_args(argv)

    insights = run(
        tenant=args.tenant,
        threshold=args.threshold,
        job_references=args.job_references,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(
            json.dumps(
                [advisory_payload(insight) for insight in insights],
                indent=2,
                default=json_default,
                allow_nan=False,
            )
        )
    else:
        for insight in insights:
            print(f"{insight.job_id}: {insight.status} ({len(insight.why_lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
