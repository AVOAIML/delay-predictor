"""Weight Agent — resolves the M3 production-delay risk-signal weight vector.

Public entry point: :class:`WeightAgent`. See ``resolver.py`` for the full
§4 resolution flow (configured -> availability -> historical blend ->
two-stage LLM-adjusted prior -> fallback) this class implements.
"""

from m3_production_delay.llm_agents.weight_agent.config import (
    DOMAIN_PRIOR_BP,
    HistoryPolicyConfig,
    WeightAgentConfig,
    get_weight_agent_config,
)
from m3_production_delay.llm_agents.weight_agent.exceptions import (
    AllSignalsUnavailableError,
    FittedWeightsError,
    LLMAdjustmentError,
    ProjectionInvariantError,
    WeightAgentError,
    WeightConfigError,
)
from m3_production_delay.llm_agents.weight_agent.history_policy import compute_usable_floor
from m3_production_delay.llm_agents.weight_agent.models import (
    CRITICAL_PROFILE_FIELDS,
    PROFILE_FIELDS,
    SIGNAL_ORDER,
    Bounds,
    ExcludedSignal,
    FittedWeights,
    HistoryAdmissibility,
    HistoryAssessment,
    HistoryInputs,
    TenantProfile,
    WeightResolution,
    bp_to_percent,
)
from m3_production_delay.llm_agents.weight_agent.profile_extractor import (
    ProfileExtractionResult,
    TenantProfileExtractor,
)
from m3_production_delay.llm_agents.weight_agent.providers import (
    FittedWeightsProvider,
    NullFittedWeightsProvider,
)
from m3_production_delay.llm_agents.weight_agent.resolver import WeightAgent
from m3_production_delay.llm_agents.weight_agent.weight_adjustment import (
    AdjustmentProposal,
    WeightAdjustmentGenerator,
)

__all__ = [
    "WeightAgent",
    "WeightAgentConfig",
    "HistoryPolicyConfig",
    "get_weight_agent_config",
    "compute_usable_floor",
    "DOMAIN_PRIOR_BP",
    "WeightAgentError",
    "WeightConfigError",
    "LLMAdjustmentError",
    "ProjectionInvariantError",
    "FittedWeightsError",
    "AllSignalsUnavailableError",
    "SIGNAL_ORDER",
    "PROFILE_FIELDS",
    "CRITICAL_PROFILE_FIELDS",
    "Bounds",
    "ExcludedSignal",
    "FittedWeights",
    "HistoryAdmissibility",
    "HistoryAssessment",
    "HistoryInputs",
    "TenantProfile",
    "WeightResolution",
    "bp_to_percent",
    "FittedWeightsProvider",
    "NullFittedWeightsProvider",
    "TenantProfileExtractor",
    "ProfileExtractionResult",
    "WeightAdjustmentGenerator",
    "AdjustmentProposal",
]
