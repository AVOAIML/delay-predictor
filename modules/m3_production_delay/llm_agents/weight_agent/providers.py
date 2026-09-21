"""DI seam for the historical fitting pipeline (spec §4.3). Not built here —
:class:`NullFittedWeightsProvider` is the only shipped implementation, so
every resolution in this codebase today falls through to lambda_bp=0. A real
provider (reading a fitted-weights table/artifact) plugs in later without any
change to blend.py or resolver.py.

Returning ``None`` means exactly "fitted weights unavailable for this
tenant" — never a placeholder result standing in for "not implemented yet".
A provider that returns non-``None`` gets its :class:`FittedWeights`
validated at construction (models.FittedWeights.__post_init__ — the same
structural invariants every other source is held to, Improvement 8), so
``source=historically_fitted``/``blended`` can only ever be emitted for a
genuinely valid result, never a stand-in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from m3_production_delay.llm_agents.weight_agent.models import FittedWeights


@runtime_checkable
class FittedWeightsProvider(Protocol):
    def get(self, tenant_id: str) -> FittedWeights | None: ...


class NullFittedWeightsProvider:
    """Always reports no fitted weights. Every resolution takes the LLM /
    fallback path until a real provider is registered."""

    def get(self, tenant_id: str) -> FittedWeights | None:
        return None
