"""Orchestrator-focused tests only (spec §18) — the Weight Agent's own
statistical/LLM logic already has its full test suite under
tests/unit/m3_production_delay/weight_agent/. These tests exist to prove the
orchestrator wires requests through unchanged and never adds, hides, or
reinterprets behaviour of its own.
"""

from __future__ import annotations

import json

import pytest

from m3_production_delay.llm_agents.weight_agent.exceptions import AllSignalsUnavailableError
from m3_production_delay.llm_agents.weight_agent.models import (
    SIGNAL_ORDER,
    SOURCE_BLENDED,
    SOURCE_CONFIGURED,
    SOURCE_HISTORICALLY_FITTED,
    SOURCE_LLM_ADJUSTED_PRIOR,
    STATUS_ACTIVE,
    TOTAL_BP,
    FittedWeights,
)
from m3_production_delay.orchestrator import ProductionDelayOrchestrator, WeightAgentRequest
from m3_production_delay.tenant_context_reader import TenantContextReader

ALL_AVAILABLE = {signal: True for signal in SIGNAL_ORDER}
NONE_AVAILABLE = {signal: False for signal in SIGNAL_ORDER}

_VALID_PROFILE_JSON = json.dumps(
    {
        "profile": {
            "production_type": "make_to_order",
            "material_dependency": "high",
            "supplier_dependency": "high",
            "workforce_dependency": "low",
            "seasonality_level": "low",
        }
    }
)
_VALID_ADJUSTMENT_JSON = json.dumps(
    {
        "adjustments_bp": {
            "time_overrun": 200,
            "operator_skill": -200,
            "seasonality": 0,
            "material_availability": 0,
            "supplier_reliability": 0,
        },
        "evidence": ["high material/supplier dependency raises time_overrun weight"],
    }
)


class _StaticJsonLLMProvider:
    """Returns one canned response per call, in order. Records every prompt
    it was given so a test can assert on call count without depending on the
    Weight Agent's internal call order beyond what's actually documented
    (profile extraction, then weight adjustment)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, max_tokens: int = 256, generation_config=None) -> str:
        self.prompts.append(prompt)
        return self._responses[len(self.prompts) - 1]


class _FixedFittedWeightsProvider:
    def __init__(self, fitted: FittedWeights) -> None:
        self._fitted = fitted

    def get(self, tenant_id: str) -> FittedWeights | None:
        return self._fitted


def _weights_sum_and_bounds_ok(result) -> bool:
    if sum(result.weights_bp.values()) != TOTAL_BP:
        return False
    return all(result.bounds_bp[s].contains(result.weights_bp[s]) for s in SIGNAL_ORDER)


# --- A. configured path -----------------------------------------------------


def test_configured_path_returns_configured_resolution_without_calling_llm():
    provider = _StaticJsonLLMProvider([])  # would raise IndexError if ever called
    orchestrator = ProductionDelayOrchestrator(llm_provider=provider)
    configured_bp = {
        "time_overrun": 4000,
        "operator_skill": 3000,
        "seasonality": 1000,
        "material_availability": 1500,
        "supplier_reliability": 500,
    }
    request = WeightAgentRequest(
        tenant_id="tenant-a", availability=ALL_AVAILABLE, configured_bp=configured_bp
    )

    result = orchestrator.resolve_weights(request)

    assert result.source == SOURCE_CONFIGURED
    assert result.status == STATUS_ACTIVE
    assert result.weights_bp == configured_bp
    assert result.requires_admin_approval is False
    assert provider.prompts == []  # configured priority means the LLM is never reached


# --- B. cold start -----------------------------------------------------------


def test_cold_start_invokes_weight_agent_llm_path():
    provider = _StaticJsonLLMProvider([_VALID_PROFILE_JSON, _VALID_ADJUSTMENT_JSON])
    orchestrator = ProductionDelayOrchestrator(llm_provider=provider)
    request = WeightAgentRequest(
        tenant_id="tenant-b",
        availability=ALL_AVAILABLE,
        tenant_description=(
            "A make-to-order metal manufacturer with high dependency on imported "
            "raw materials and suppliers, stable workforce, low seasonal variation."
        ),
    )

    result = orchestrator.resolve_weights(request)

    assert result.source == SOURCE_LLM_ADJUSTED_PRIOR
    assert result.requires_admin_approval is True
    assert _weights_sum_and_bounds_ok(result)
    assert len(provider.prompts) == 2  # stage 1 (profile) then stage 2 (adjustment)


# --- C. all signals unavailable ---------------------------------------------


def test_all_signals_unavailable_raises_and_is_not_converted_to_a_result():
    orchestrator = ProductionDelayOrchestrator()
    request = WeightAgentRequest(
        tenant_id="tenant-c",
        availability=NONE_AVAILABLE,
        exclusion_reasons={s: "no_computable_signals" for s in SIGNAL_ORDER},
    )

    with pytest.raises(AllSignalsUnavailableError):
        orchestrator.resolve_weights(request)


# --- D. unexpected exception propagation ------------------------------------


def test_unexpected_exception_is_not_swallowed_into_a_successful_result():
    orchestrator = ProductionDelayOrchestrator()
    # A malformed `availability` (not the dict[str, bool] the contract requires)
    # is a caller bug, not a Weight Agent domain error — nothing in resolve()
    # catches it. The orchestrator must not catch it either.
    request = WeightAgentRequest(tenant_id="tenant-d", availability=["not", "a", "dict"])  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        orchestrator.resolve_weights(request)


# --- E. fitted path -----------------------------------------------------------


def test_fitted_provider_is_used_and_projected_into_bounds():
    fitted = FittedWeights(
        weights_bp={signal: 2000 for signal in SIGNAL_ORDER},
        signal_set=frozenset(SIGNAL_ORDER),
        delayed_event_count=500,
    )
    orchestrator = ProductionDelayOrchestrator(fitted_provider=_FixedFittedWeightsProvider(fitted))
    request = WeightAgentRequest(tenant_id="tenant-e", availability=ALL_AVAILABLE)

    result = orchestrator.resolve_weights(request)

    assert result.source in (SOURCE_BLENDED, SOURCE_HISTORICALLY_FITTED)
    assert _weights_sum_and_bounds_ok(result)
    # The even 2000-bp split violates this tenant's configured Bounds
    # (time_overrun min=3000, supplier_reliability max=1500) — the orchestrator
    # must surface that the Weight Agent projected it, not hide the fact.
    assert result.fitted_projection_applied is True
    assert result.fitted_projection_changed_signals


# --- F. tenant identity -------------------------------------------------------


def test_tenant_identity_passes_through_unchanged_and_llm_cannot_override_it():
    injected_tenant_id = json.dumps(
        {
            "tenant_id": "attacker-supplied-tenant",
            "profile": {
                "production_type": "make_to_stock",
                "material_dependency": "low",
                "supplier_dependency": "low",
                "workforce_dependency": "low",
                "seasonality_level": "none",
            },
        }
    )
    provider = _StaticJsonLLMProvider([injected_tenant_id, _VALID_ADJUSTMENT_JSON])
    orchestrator = ProductionDelayOrchestrator(llm_provider=provider)
    request = WeightAgentRequest(
        tenant_id="tenant-f-trusted",
        availability=ALL_AVAILABLE,
        tenant_description="Any description text.",
    )

    result = orchestrator.resolve_weights(request)

    assert result.tenant_id == "tenant-f-trusted"


# --- G. TenantContextReader integration --------------------------------------


def test_resolved_tenant_description_is_passed_unchanged_to_weight_agent():
    """When the caller supplies no tenant_description, the orchestrator must
    resolve one via TenantContextReader and hand it to WeightAgent.resolve()
    unchanged — proven here by checking the resolved text reaches the actual
    LLM prompt verbatim, not by re-testing profile extraction itself."""
    resolved_text = "Resolved-via-metadata-source description for integration test."

    class _FixedMetadataSource:
        def get_description(self, tenant_id: str) -> str | None:
            return resolved_text

    reader = TenantContextReader(metadata_source=_FixedMetadataSource())
    provider = _StaticJsonLLMProvider([_VALID_PROFILE_JSON, _VALID_ADJUSTMENT_JSON])
    orchestrator = ProductionDelayOrchestrator(llm_provider=provider, tenant_context_reader=reader)
    request = WeightAgentRequest(tenant_id="tenant-g", availability=ALL_AVAILABLE)

    orchestrator.resolve_weights(request)

    assert provider.prompts, "expected the LLM to be called at all"
    assert resolved_text in provider.prompts[0]


def test_explicit_tenant_description_bypasses_tenant_context_reader():
    class _ExplodingMetadataSource:
        def get_description(self, tenant_id: str) -> str | None:
            raise AssertionError("TenantContextReader must not be consulted")

    reader = TenantContextReader(metadata_source=_ExplodingMetadataSource())
    provider = _StaticJsonLLMProvider([_VALID_PROFILE_JSON, _VALID_ADJUSTMENT_JSON])
    orchestrator = ProductionDelayOrchestrator(llm_provider=provider, tenant_context_reader=reader)
    request = WeightAgentRequest(
        tenant_id="tenant-h",
        availability=ALL_AVAILABLE,
        tenant_description="explicitly supplied by an existing trusted caller",
    )

    orchestrator.resolve_weights(request)  # must not raise via _ExplodingMetadataSource

    assert "explicitly supplied by an existing trusted caller" in provider.prompts[0]
