"""The judge evaluation set and its harness.

The set itself is checked offline, every run: it must be well-formed, balanced,
and — crucially — every "supported" case must be one this pipeline's own
deterministic validators accept. A case the validators would reject could
never reach the judge in production, so scoring the judge on it would measure
nothing.

The real-LLM run is a separate, opt-in test, because it costs money and needs
a configured provider.
"""

from __future__ import annotations

import json
import os

import pytest

from conftest import agent_for, approving_agent, rejecting_agent, unparsable_agent
from m3_production_delay.review.eval.run_judge_eval import (
    CASES_PATH,
    ENV_FLAG,
    load_cases,
    report,
    run_cases,
    score,
)
from m3_production_delay.review.schemas import EvidencePack, InsightDraft, InsightLine, has_errors
from m3_production_delay.review.validators import validate

CATEGORIES = {"supported", "fabricated_cause", "wrong_number", "non_fired_signal", "omitted_signal"}


@pytest.fixture(scope="module")
def cases() -> dict:
    return load_cases()


# ─── the set ─────────────────────────────────────────────────────────────────


def test_the_set_is_balanced(cases):
    assert len(cases["cases"]) == 20
    supported = [c for c in cases["cases"] if c["expected_approved"]]
    assert len(supported) == 10
    assert {c["category"] for c in cases["cases"]} == CATEGORIES


def test_every_unsupported_category_is_represented(cases):
    counts: dict[str, int] = {}
    for case in cases["cases"]:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert counts["fabricated_cause"] == 3
    assert counts["wrong_number"] == 3
    assert counts["non_fired_signal"] == 2
    assert counts["omitted_signal"] == 2


def test_case_ids_are_unique_and_every_scenario_resolves(cases):
    ids = [case["id"] for case in cases["cases"]]
    assert len(ids) == len(set(ids))
    for case in cases["cases"]:
        assert case["scenario"] in cases["scenarios"]
        assert case["lines"], f"{case['id']} has no candidate lines"


def test_every_scenario_is_a_real_evidence_pack(cases):
    for name, scenario in cases["scenarios"].items():
        pack = EvidencePack.from_dict(scenario)
        assert pack.job_id
        assert pack.operations, name
        assert pack.delay_threshold == cases["threshold"]


def test_every_supported_case_would_pass_our_own_validators(cases):
    """The ground truth has to agree with the deterministic layer: a
    "supported" case the validators would reject is not a judge failure
    waiting to happen, it is a broken test case."""
    for case in cases["cases"]:
        if not case["expected_approved"]:
            continue
        pack = EvidencePack.from_dict(cases["scenarios"][case["scenario"]])
        draft = InsightDraft(
            job_id=pack.job_id,
            summary_overrun_hours=pack.summary_overrun_hours,
            summary_risk_score=pack.summary_risk_score,
            summary_is_delayed=pack.summary_is_delayed,
            summary_basis=pack.summary_basis,
            why_lines=tuple(InsightLine.from_dict(line) for line in case["lines"]),
            material_overrun=pack.material_overrun,
        )
        assert not has_errors(validate(pack, draft)), case["id"]


def test_every_unsupported_case_differs_from_the_supported_baseline(cases):
    baseline = {
        case["scenario"]: json.dumps(case["lines"], sort_keys=True)
        for case in cases["cases"]
        if case["expected_approved"]
    }
    for case in cases["cases"]:
        if case["expected_approved"]:
            continue
        assert json.dumps(case["lines"], sort_keys=True) != baseline.get(case["scenario"]), (
            f"{case['id']} is identical to a supported case"
        )


def test_the_file_is_regenerable_and_committed(cases):
    assert CASES_PATH.exists()
    assert cases["version"] == 1


# ─── the harness ─────────────────────────────────────────────────────────────


def test_a_perfect_judge_scores_one(cases):
    """Validates the scorer, not a model: a judge that answers every case
    correctly must come out at precision and recall 1.0."""

    # Answers from the recorded ground truth, in file order — `run_cases`
    # walks the cases in order and makes exactly one call each. Identifying
    # the case from the prompt text instead would not work: some unsupported
    # variants (a line moved to an operation where its signal never fired)
    # render identically to the supported baseline, which is precisely the
    # kind of case a judge has to read the evidence to catch.
    pending = iter(cases["cases"])

    def _approve(prompt: str) -> str:
        count = prompt.count("] signal=")
        return json.dumps(
            {
                "approved": True,
                "lines": [{"line_index": i, "supported": True} for i in range(count)],
                "unsupported_claims": [],
                "omitted_signals": [],
            }
        )

    def _refuse(prompt: str, target: int) -> str:
        count = prompt.count("] signal=")
        return json.dumps(
            {
                "approved": False,
                "lines": [
                    {"line_index": i, "supported": i != target} for i in range(count)
                ],
                "unsupported_claims": ["not supported"],
                "omitted_signals": [],
            }
        )

    def oracle(prompt: str) -> str:
        case = next(pending)
        assert prompt.count("] signal=") == len(case["lines"]), case["id"]
        if case["expected_approved"]:
            return _approve(prompt)
        target = case.get("target_index")
        if target is None:  # omission: every line is fine, the set is not
            verdict = json.loads(_approve(prompt))
            verdict["approved"] = False
            verdict["omitted_signals"] = ["operator_pace_ratio"]
            return json.dumps(verdict)
        return _refuse(prompt, target)

    metrics = score(run_cases(agent_for(oracle), cases))
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["parse_failures"] == 0


def test_a_judge_that_refuses_everything_has_perfect_recall_and_poor_precision(cases):
    metrics = score(run_cases(rejecting_agent(0), cases))
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["false_positives"] == 10


def test_an_unusable_provider_shows_up_as_parse_failures(cases):
    results = run_cases(unparsable_agent(), cases)
    metrics = score(results)
    assert metrics["parse_failures"] == 20
    # Everything is refused, so recall is trivially perfect — which is
    # exactly why parse_failures is reported alongside it.
    assert metrics["recall"] == 1.0
    assert "parse_error" in report(results)


def test_the_report_names_every_case(cases):
    text = report(run_cases(approving_agent(), cases))
    for case in cases["cases"]:
        assert case["id"] in text
    assert "precision" in text and "pinpoint accuracy" in text


# ─── the real thing, opt-in ──────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get(ENV_FLAG),
    reason=f"set {ENV_FLAG}=1 (with a configured LLM_PROVIDER) to score a real judge",
)
def test_real_llm_judge_evaluation(cases):
    from m3_production_delay.llm_agents.review_agent import ReviewAgent

    results = run_cases(ReviewAgent(), cases)
    print(report(results))
    metrics = score(results)
    # Deliberately loose: this exists to produce the numbers, not to gate a
    # build on a hosted model's behaviour on a given day.
    assert metrics["cases"] == 20
