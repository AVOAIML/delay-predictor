import json

import pytest

from m3_production_delay.llm_agents.weight_agent.config import build_config
from m3_production_delay.llm_agents.weight_agent.exceptions import LLMAdjustmentError
from m3_production_delay.llm_agents.weight_agent.models import Bounds, TenantProfile
from m3_production_delay.llm_agents.weight_agent.weight_adjustment import WeightAdjustmentGenerator

CONFIG = build_config()
PROFILE = TenantProfile(
    industry="fabrication",
    production_type="make_to_order",
    material_dependency="high",
    supplier_dependency="high",
    workforce_dependency="medium",
    workforce_stability="stable",
    seasonality_level="low",
    automation_level="manual",
    make_to_order_ratio="high",
    supply_chain_complexity="complex",
)
PRIOR_BP = {
    "time_overrun": 4000,
    "operator_skill": 3500,
    "seasonality": 1000,
    "material_availability": 1000,
    "supplier_reliability": 500,
}
BOUNDS_BP = {
    "time_overrun": Bounds(3000, 5000),
    "operator_skill": Bounds(2000, 4500),
    "seasonality": Bounds(500, 2000),
    "material_availability": Bounds(500, 2500),
    "supplier_reliability": Bounds(300, 1500),
}
ZERO_SUM_ADJUSTMENT = {
    "time_overrun": 100,
    "operator_skill": -100,
    "seasonality": 0,
    "material_availability": 0,
    "supplier_reliability": 0,
}


class ScriptedProvider:
    name = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, *, max_tokens: int = 256, generation_config=None) -> str:
        self.calls.append(prompt)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _valid_response(**overrides) -> str:
    payload = {
        "adjustments_bp": dict(ZERO_SUM_ADJUSTMENT),
        "evidence": ["High material dependency implies more overrun weight"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _propose(provider, profile=PROFILE):
    return WeightAdjustmentGenerator(provider).propose(profile, PRIOR_BP, BOUNDS_BP, CONFIG)


def test_valid_response_parsed_on_first_attempt():
    provider = ScriptedProvider([_valid_response()])
    result = _propose(provider)
    assert result.adjustment_bp == ZERO_SUM_ADJUSTMENT
    assert result.retried is False
    assert len(provider.calls) == 1


def test_json_code_fence_is_unwrapped_before_parsing():
    provider = ScriptedProvider([f"```json\n{_valid_response()}\n```"])
    result = _propose(provider)
    assert result.adjustment_bp == ZERO_SUM_ADJUSTMENT
    assert result.retried is False


def test_prompt_never_contains_raw_tenant_description_text():
    # the trust boundary from Improvement 2: this stage's signature has no
    # description parameter at all, so there is nothing to check for leakage
    # other than confirming the prompt is built purely from the structured
    # profile and numeric context.
    provider = ScriptedProvider([_valid_response()])
    _propose(provider)
    prompt = provider.calls[0]
    assert "tenant_description" not in prompt.lower()
    assert "industry: fabrication" in prompt.lower()


def test_prompt_contains_domain_guidance_and_rejects_unjustified_zero_adjustment():
    provider = ScriptedProvider([_valid_response()])
    _propose(provider)
    prompt = provider.calls[0]
    normalized_prompt = " ".join(prompt.lower().split())

    assert "high material dependency should generally increase material_availability" in normalized_prompt
    assert "unstable suppliers should generally increase" in normalized_prompt
    assert "supplier_reliability weight" in normalized_prompt
    assert "the current prior is a baseline, not a preferred answer" in normalized_prompt
    assert "being within bounds is not by itself a reason to return zero adjustment" in normalized_prompt
    assert "zero adjustment is allowed only when" in normalized_prompt


def test_invalid_json_falls_back_after_one_retry():
    provider = ScriptedProvider(["not json", "still not json"])
    with pytest.raises(LLMAdjustmentError, match="invalid_json"):
        _propose(provider)
    assert len(provider.calls) == 2


def test_schema_mismatch_missing_adjustments_key():
    response = json.dumps({"evidence": []})
    provider = ScriptedProvider([response, response])
    with pytest.raises(LLMAdjustmentError, match="schema_mismatch"):
        _propose(provider)


def test_out_of_enum_signal_key_falls_back():
    bad = dict(ZERO_SUM_ADJUSTMENT)
    del bad["seasonality"]
    bad["not_a_real_signal"] = 0
    response = _valid_response(adjustments_bp=bad)
    provider = ScriptedProvider([response, response])
    with pytest.raises(LLMAdjustmentError, match="out_of_enum_signal"):
        _propose(provider)


def test_non_zero_sum_adjustment_is_rejected_not_normalized():
    bad = dict(ZERO_SUM_ADJUSTMENT)
    bad["time_overrun"] += 5
    response = _valid_response(adjustments_bp=bad)
    provider = ScriptedProvider([response, response])
    with pytest.raises(LLMAdjustmentError, match="non_zero_sum_adjustment"):
        _propose(provider)


def test_retry_succeeds_after_one_bad_attempt():
    provider = ScriptedProvider(["not json", _valid_response()])
    result = _propose(provider)
    assert result.retried is True
    assert len(provider.calls) == 2


def test_tenant_identity_in_response_is_ignored():
    payload = json.loads(_valid_response())
    payload["tenant_id"] = "some-other-tenant"
    payload["weights_bp"] = {s: 2000 for s in ZERO_SUM_ADJUSTMENT}
    provider = ScriptedProvider([json.dumps(payload)])
    result = _propose(provider)
    assert result.adjustment_bp == ZERO_SUM_ADJUSTMENT
    assert not hasattr(result, "tenant_id")
    assert not hasattr(result, "weights_bp")


def test_aggressive_but_schema_valid_adjustment_is_accepted_here_bounds_enforced_later():
    # This stage validates structure (keys, ints, sum==0) only; whether the
    # magnitude is sane against bounds is projection.py's job, tested in
    # test_projection.py and exercised end-to-end in test_resolver.py's
    # adversarial case. Confirms this stage doesn't itself clip or reject on
    # magnitude alone.
    aggressive = {
        "time_overrun": 4000,
        "operator_skill": -4000,
        "seasonality": 3000,
        "material_availability": 2000,
        "supplier_reliability": -5000,
    }
    assert sum(aggressive.values()) == 0
    provider = ScriptedProvider([_valid_response(adjustments_bp=aggressive)])
    result = _propose(provider)
    assert result.adjustment_bp == aggressive
