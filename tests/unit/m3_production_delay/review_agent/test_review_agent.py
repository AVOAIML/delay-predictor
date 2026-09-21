"""The Review Agent itself: its config, what it sends, and the fact that every
failure path produces an unapproved verdict rather than an exception or a
default yes."""

from __future__ import annotations

import json

import pytest

from conftest import ScriptedProvider
from m3_production_delay.llm_agents.review_agent import (
    PROMPT_VERSION,
    JudgeVerdict,
    ReviewAgent,
    ReviewAgentConfig,
    ReviewAgentTracer,
    ReviewConfigError,
    build_config,
    get_review_agent_config,
)
from m3_production_delay.llm_agents.review_agent.models import coerce_diagnostics


def _approve(count: int, **overrides) -> str:
    payload = {
        "approved": True,
        "lines": [{"line_index": i, "supported": True} for i in range(count)],
        "unsupported_claims": [],
        "omitted_signals": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _agent(response: str, **config_overrides) -> tuple[ReviewAgent, ScriptedProvider]:
    provider = ScriptedProvider(response)
    return (
        ReviewAgent(llm_provider=provider, config=build_config(**config_overrides)),
        provider,
    )


# ─── config ──────────────────────────────────────────────────────────────────


def test_the_shipped_config_is_the_v1_contract():
    config = get_review_agent_config()
    assert config.max_tokens == 900
    assert config.max_attempts == 2
    assert config.temperature == 0.0
    assert config.seed == 0


def test_the_config_is_a_cached_singleton():
    assert get_review_agent_config() is get_review_agent_config()


def test_llm_enabled_comes_from_settings_not_the_environment(monkeypatch):
    from maxxflow_core.settings import get_settings

    monkeypatch.setenv("M3_REVIEW_AGENT_LLM_ENABLED", "false")
    get_settings.cache_clear()
    get_review_agent_config.cache_clear()
    try:
        assert get_review_agent_config().llm_enabled is False
    finally:
        get_settings.cache_clear()
        get_review_agent_config.cache_clear()


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_tokens", 0), ("max_attempts", 0), ("temperature", 3.0)],
)
def test_an_unusable_parameter_is_refused_at_construction(field, value):
    with pytest.raises(ReviewConfigError):
        ReviewAgentConfig(**{field: value})


def test_a_zero_attempt_budget_names_what_it_would_cost():
    with pytest.raises(ReviewConfigError, match="publish every insight unjudged"):
        ReviewAgentConfig(max_attempts=0)


# ─── what the agent sends ────────────────────────────────────────────────────


def test_one_call_carries_the_evidence_the_baselines_and_the_indexed_lines(pack, draft):
    agent, provider = _agent(_approve(len(draft.why_lines)))
    agent.judge(pack, draft)

    assert len(provider.calls) == 1, "the agent makes exactly one call per round"
    prompt = provider.calls[0]
    assert "EVIDENCE PACK" in prompt
    assert "FIRE BASELINES" in prompt
    assert pack.job_id in prompt
    for line in draft.why_lines:
        assert f"[{line.index}] signal={line.signal_key}" in prompt


def test_the_prompt_states_the_same_baselines_the_composer_used(pack, draft):
    from m3_production_delay.review.evidence import FIRE_BASELINES

    agent, provider = _agent(_approve(len(draft.why_lines)))
    agent.judge(pack, draft)

    # Borrowed from the deterministic package rather than restated, so the
    # judge can never be told a threshold the composer did not apply.
    for key, value in FIRE_BASELINES.items():
        assert f"{key}: > {value}" in provider.calls[0]


def test_generation_is_pinned_for_reproducibility(pack, draft):
    agent, provider = _agent(_approve(len(draft.why_lines)))
    agent.judge(pack, draft)

    assert provider.max_tokens == [agent.config.max_tokens]
    config = provider.generation_configs[0]
    assert config.temperature == 0.0
    assert config.seed == 0


def test_the_agent_never_sends_an_infinity(pack, draft):
    agent, provider = _agent(_approve(len(draft.why_lines)))
    agent.judge(pack, draft)
    assert "Infinity" not in provider.calls[0]


# ─── verdicts ────────────────────────────────────────────────────────────────


def test_an_approved_verdict_is_returned_as_given(pack, draft):
    agent, _ = _agent(_approve(len(draft.why_lines)))
    verdict = agent.judge(pack, draft)

    assert verdict.approved is True
    assert len(verdict.lines) == len(draft.why_lines)
    assert verdict.parse_error is None


def test_an_empty_draft_is_never_sent(pack, draft):
    agent, provider = _agent(_approve(0))
    verdict = agent.judge(pack, draft.with_lines(()))

    assert verdict.approved is True and verdict.skipped is True
    assert provider.calls == [], "nothing to judge, so nothing to ask"


def test_a_disabled_agent_degrades_like_an_unreachable_one(pack, draft):
    agent, provider = _agent(_approve(len(draft.why_lines)), llm_enabled=False)
    verdict = agent.judge(pack, draft)

    assert verdict.approved is False
    assert verdict.skipped is True
    assert verdict.parse_error == "llm_disabled"
    assert provider.calls == []


def test_a_provider_failure_is_reported_not_raised(pack, draft):
    class Exploding(ScriptedProvider):
        def generate(self, prompt, *, max_tokens=256, generation_config=None):
            raise TimeoutError("provider timed out")

    agent = ReviewAgent(llm_provider=Exploding(), config=build_config())
    verdict = agent.judge(pack, draft)

    assert verdict.approved is False
    assert verdict.parse_error.startswith("provider_error")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("not json", "invalid_json"),
        ("[]", "schema_mismatch"),
        ('{"lines": []}', "schema_mismatch"),
        ('{"approved": true, "lines": {}}', "schema_mismatch"),
        ('{"approved": true, "lines": [{"line_index": 0, "supported": true}]}',
         "incomplete_verdict"),
        ('{"approved": true, "lines": [{"line_index": 99, "supported": true}]}', "out_of_range"),
    ],
)
def test_no_malformed_response_is_ever_read_as_approval(pack, draft, response, expected):
    agent, _ = _agent(response)
    verdict = agent.judge(pack, draft)

    assert verdict.approved is False
    assert expected in verdict.parse_error


def test_a_self_contradicting_approval_is_read_conservatively(pack, draft):
    count = len(draft.why_lines)
    response = json.dumps(
        {
            "approved": True,
            "lines": [{"line_index": i, "supported": i != 1} for i in range(count)],
        }
    )
    agent, _ = _agent(response)
    verdict = agent.judge(pack, draft)

    assert verdict.approved is False
    assert verdict.unsupported_indices == (1,)


def test_a_claimed_omission_blocks_approval(pack, draft):
    agent, _ = _agent(
        _approve(len(draft.why_lines), omitted_signals=["material_shortfall_ratio"])
    )
    verdict = agent.judge(pack, draft)

    assert verdict.approved is False
    assert verdict.omitted_signals == ("material_shortfall_ratio",)


# ─── the verdict type's own guarantees ───────────────────────────────────────


def test_a_failed_parse_cannot_be_constructed_as_approved():
    with pytest.raises(ValueError, match="cannot be approved"):
        JudgeVerdict(approved=True, parse_error="invalid_json: ...")


def test_diagnostics_are_capped_in_count_and_length():
    long_claims = ["x" * 500] * 50
    assert len(coerce_diagnostics(long_claims)) == 20
    assert all(len(item) == 300 for item in coerce_diagnostics(long_claims))
    # Anything that is not a list of scalars contributes nothing.
    assert coerce_diagnostics("not a list") == ()
    assert coerce_diagnostics([{"nested": 1}]) == ()


def test_prose_cannot_be_smuggled_through_a_line_verdict(pack, draft):
    response = json.dumps(
        {
            "approved": False,
            "lines": [
                {
                    "line_index": i,
                    "supported": False,
                    "evidence_ref": "e" * 1000,
                    "issue": "i" * 1000,
                }
                for i in range(len(draft.why_lines))
            ],
        }
    )
    agent, _ = _agent(response)
    verdict = agent.judge(pack, draft)

    assert all(len(line.evidence_ref) == 300 for line in verdict.lines)
    assert all(len(line.issue) == 300 for line in verdict.lines)


# ─── tracing ─────────────────────────────────────────────────────────────────


def test_tracing_is_off_by_default_and_content_is_gated_separately(pack, draft, caplog):
    agent, _ = _agent(_approve(len(draft.why_lines)))
    agent.judge(pack, draft)  # NOOP_TRACER
    assert "REVIEW TRACE" not in caplog.text


def test_content_tracing_requires_both_switches(pack, draft):
    metadata_only = ReviewAgentTracer(trace_id="t", enabled=True, include_content=False)
    assert metadata_only.include_content is False
    # The second switch cannot turn content on by itself.
    content_without_trace = ReviewAgentTracer(trace_id="t", enabled=False, include_content=True)
    assert content_without_trace.include_content is False


def test_an_enabled_trace_reports_the_verdict_without_the_prompt(pack, draft, caplog):
    import logging

    tracer = ReviewAgentTracer(trace_id="trace-1", enabled=True, include_content=False)
    agent, _ = _agent(_approve(len(draft.why_lines)))
    with caplog.at_level(logging.INFO, logger="m3_production_delay.review_agent.trace"):
        agent.judge(pack, draft, tracer=tracer)

    assert "M3 REVIEW AGENT VERDICT" in caplog.text
    assert "trace-1" in caplog.text
    assert PROMPT_VERSION in caplog.text
    # Content is gated by the second switch, so no prompt or response body.
    assert "PROMPT START" not in caplog.text
    assert "EVIDENCE PACK" not in caplog.text


# ─── markdown-fenced responses ───────────────────────────────────────────────
#
# Verified against a real Azure AI Foundry deployment: the model answers with a
# ```json fence despite the prompt asking for JSON only. The content inside is
# exactly the contract, so the fence is unwrapped rather than treated as a
# parse failure — but nothing else is relaxed.


@pytest.mark.parametrize(
    ("wrapper", "label"),
    [
        ("```json\n{body}\n```", "json-tagged fence"),
        ("```\n{body}\n```", "bare fence"),
        ("  ```json\n{body}\n```  ", "fence with surrounding whitespace"),
        ("{body}", "no fence at all"),
    ],
)
def test_a_fenced_verdict_is_still_read(pack, draft, wrapper, label):
    inner = _approve(len(draft.why_lines))
    agent, _ = _agent(wrapper.replace("{body}", inner))
    verdict = agent.judge(pack, draft)

    assert verdict.approved is True, label
    assert verdict.parse_error is None, label
    assert len(verdict.lines) == len(draft.why_lines)


def test_unwrapping_the_fence_does_not_relax_anything_else(pack, draft):
    # A fenced response that is valid JSON but the wrong shape must still fail:
    # the fence is packaging, not permission.
    agent, _ = _agent('```json\n{"approved": "yes", "lines": []}\n```')
    verdict = agent.judge(pack, draft)

    assert verdict.approved is False
    assert "schema_mismatch" in verdict.parse_error


def test_prose_outside_a_fence_is_still_a_parse_failure(pack, draft):
    agent, _ = _agent("Here is my verdict:\n\n" + _approve(len(draft.why_lines)))
    verdict = agent.judge(pack, draft)

    assert verdict.approved is False
    assert "invalid_json" in verdict.parse_error


def test_strip_code_fence_in_isolation():
    from m3_production_delay.llm_agents.review_agent.resolver import strip_code_fence

    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'
    # An unterminated fence still yields its content rather than nothing.
    assert strip_code_fence('```json\n{"a": 1}') == '{"a": 1}'


def test_commentary_after_the_verdict_is_ignored(pack, draft):
    """Also from the real deployment: the model appends a sentence after the
    JSON, which json.loads rejects as "Extra data". The verdict itself is
    complete, so it is read and the trailing prose dropped."""
    agent, _ = _agent(
        _approve(len(draft.why_lines))
        + "\n\nAll four lines are supported by the evidence pack above."
    )
    verdict = agent.judge(pack, draft)

    assert verdict.approved is True
    assert verdict.parse_error is None


def test_a_fenced_verdict_with_trailing_commentary_is_read(pack, draft):
    agent, _ = _agent(
        "```json\n" + _approve(len(draft.why_lines)) + "\n```\n\nHope that helps."
    )
    verdict = agent.judge(pack, draft)
    assert verdict.approved is True


def test_a_truncated_verdict_still_fails(pack, draft):
    # max_tokens cut the object off mid-way: there is no complete document to
    # read, so this must not be salvaged into a partial verdict.
    agent, _ = _agent('{"approved": true, "lines": [{"line_index": 0, "suppo')
    verdict = agent.judge(pack, draft)

    assert verdict.approved is False
    assert "invalid_json" in verdict.parse_error


# ─── the re-judge has to be answerable ───────────────────────────────────────


def test_the_re_judge_is_told_what_it_already_withdrew(pack, draft):
    """Without this the retry cannot succeed. Every composed line is for a
    fired, weighted signal, so dropping one at the judge's request violates its
    own "nothing omitted" check — and the second verdict then refuses the draft
    for an absence it caused. Observed against a real deployment."""
    agent, provider = _agent(_approve(len(draft.why_lines) - 1))
    agent.judge(
        pack, draft.with_lines(draft.why_lines[:-1]), withdrawn=("supplier_reliability",)
    )

    prompt = provider.calls[0]
    assert "WITHDRAWN AT YOUR REQUEST" in prompt
    assert "supplier_reliability" in prompt
    assert "NOT an omission" in prompt


def test_the_first_pass_carries_no_withdrawal_note(pack, draft):
    agent, provider = _agent(_approve(len(draft.why_lines)))
    agent.judge(pack, draft)
    assert "WITHDRAWN" not in provider.calls[0]
