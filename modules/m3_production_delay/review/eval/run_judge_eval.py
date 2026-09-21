"""Runs ``judge_cases.json`` through a real LLM and prints how well the judge
separates supported explanations from unsupported ones.

This is the only place in Section 3 that talks to a paid API, and it never
runs unless it is asked to:

    M3_REVIEW_JUDGE_EVAL=1 LLM_PROVIDER=openai LLM_MODEL_ID=<verified-id> \
      uv run python modules/m3_production_delay/review/eval/run_judge_eval.py

Without ``M3_REVIEW_JUDGE_EVAL`` set it exits immediately, so it is safe to
have in a repo whose whole test suite is offline by default.

What the numbers mean — the task scored is "flag an insight that should not
ship", so a *positive* is the judge withholding approval:

  precision  of the insights it refused, how many really were unsound.
             Low precision means it blocks good explanations, which in this
             pipeline costs a ``fallback_template`` (the templates still
             ship, just unjudged).
  recall     of the unsound insights, how many it refused. Low recall is the
             expensive one: an unsupported line reaching an operator.
  pinpoint   when it correctly refused, did it name the line that was
             actually wrong — the number that decides whether the
             drop-and-re-judge retry can recover a good explanation from a
             partly bad one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from m3_production_delay.review.judge import LLMCallable, build_llm, judge_draft
from m3_production_delay.review.schemas import EvidencePack, InsightDraft, InsightLine

CASES_PATH = Path(__file__).resolve().parent / "judge_cases.json"
ENV_FLAG = "M3_REVIEW_JUDGE_EVAL"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    expected_approved: bool
    approved: bool
    pinpointed: bool | None
    parse_error: str | None

    @property
    def flagged(self) -> bool:
        """The judge withheld approval — the positive class."""
        return not self.approved

    @property
    def should_flag(self) -> bool:
        return not self.expected_approved


def load_cases(path: Path = CASES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _draft_for(case: dict, pack: EvidencePack) -> InsightDraft:
    lines = tuple(InsightLine.from_dict(line) for line in case["lines"])
    return InsightDraft(
        job_id=pack.job_id,
        summary_overrun_hours=pack.summary_overrun_hours,
        summary_risk_score=pack.summary_risk_score,
        summary_is_delayed=pack.summary_is_delayed,
        summary_basis=pack.summary_basis,
        why_lines=lines,
        material_overrun=pack.material_overrun,
    )


def run_cases(llm: LLMCallable, payload: dict | None = None) -> list[CaseResult]:
    payload = payload or load_cases()
    packs = {
        name: EvidencePack.from_dict(scenario)
        for name, scenario in payload["scenarios"].items()
    }
    results: list[CaseResult] = []
    for case in payload["cases"]:
        pack = packs[case["scenario"]]
        verdict = judge_draft(pack, _draft_for(case, pack), llm)
        target = case.get("target_index")
        pinpointed: bool | None = None
        if target is not None and not verdict.approved:
            pinpointed = target in verdict.unsupported_indices
        results.append(
            CaseResult(
                case_id=case["id"],
                category=case["category"],
                expected_approved=bool(case["expected_approved"]),
                approved=verdict.approved,
                pinpointed=pinpointed,
                parse_error=verdict.parse_error,
            )
        )
    return results


def score(results: list[CaseResult]) -> dict:
    true_positives = sum(1 for r in results if r.flagged and r.should_flag)
    false_positives = sum(1 for r in results if r.flagged and not r.should_flag)
    false_negatives = sum(1 for r in results if not r.flagged and r.should_flag)
    precision = true_positives / (true_positives + false_positives) if (
        true_positives + false_positives
    ) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (
        true_positives + false_negatives
    ) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pinpointable = [r for r in results if r.pinpointed is not None]
    return {
        "cases": len(results),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": sum(1 for r in results if r.flagged == r.should_flag) / len(results),
        "pinpoint_accuracy": (
            sum(1 for r in pinpointable if r.pinpointed) / len(pinpointable)
            if pinpointable
            else None
        ),
        "parse_failures": sum(1 for r in results if r.parse_error),
    }


def report(results: list[CaseResult]) -> str:
    metrics = score(results)
    lines = ["", "=== M3 review judge evaluation ===", ""]
    for result in results:
        mark = "ok  " if result.flagged == result.should_flag else "MISS"
        pinpoint = "" if result.pinpointed is None else f"  pinpoint={result.pinpointed}"
        error = f"  parse_error={result.parse_error}" if result.parse_error else ""
        lines.append(
            f"  [{mark}] {result.case_id:<42} {result.category:<18} "
            f"approved={str(result.approved):<5}{pinpoint}{error}"
        )

    by_category: dict[str, list[CaseResult]] = {}
    for result in results:
        by_category.setdefault(result.category, []).append(result)
    lines += ["", "  by category:"]
    for category, group in sorted(by_category.items()):
        correct = sum(1 for r in group if r.flagged == r.should_flag)
        lines.append(f"    {category:<18} {correct}/{len(group)}")

    lines += [
        "",
        f"  precision         {metrics['precision']:.3f}   "
        "(of the insights it refused, how many were really unsound)",
        f"  recall            {metrics['recall']:.3f}   "
        "(of the unsound insights, how many it refused)",
        f"  f1                {metrics['f1']:.3f}",
        f"  accuracy          {metrics['accuracy']:.3f}",
        f"  pinpoint accuracy "
        + (
            "   n/a"
            if metrics["pinpoint_accuracy"] is None
            else f"{metrics['pinpoint_accuracy']:.3f}"
        )
        + "   (named the line that was actually wrong)",
        f"  parse failures    {metrics['parse_failures']}",
        "",
    ]
    return "\n".join(lines)


def main(llm: Callable[[str, str], str] | None = None) -> int:
    if not os.environ.get(ENV_FLAG):
        print(
            f"{ENV_FLAG} is not set — skipping. This evaluation calls a real LLM; set "
            f"{ENV_FLAG}=1 along with LLM_PROVIDER / LLM_MODEL_ID to run it."
        )
        return 0
    results = run_cases(llm or build_llm())
    print(report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
