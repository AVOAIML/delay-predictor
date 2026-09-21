"""Single source of truth for the Weight Agent's prior, bounds and policy
(spec §3). Anything that reads a weight prior or bound — this agent today,
``services/configurator`` later — imports :func:`get_weight_agent_config`
rather than holding its own copy.

Mirrors ``maxxflow_core.settings.get_settings``: a cached singleton built from
typed constants, validated once at construction (in ``__post_init__``, not a
separate function a caller could forget to invoke — see models.py's module
docstring for why every domain object here follows that rule).
``get_weight_agent_config.cache_clear()`` reloads in tests.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from maxxflow_core.settings import get_settings

from m3_production_delay.llm_agents.weight_agent.exceptions import WeightConfigError
from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER, SIGNAL_SET, TOTAL_BP, Bounds

# --- prior policy -----------------------------------------------------------
#
# DOMAIN_PRIOR_BP is a PRODUCT DECISION, not a statistical fit: the values
# from the signed user story TA/PP/PO 10.3.1. It is shipped as the active
# default below because it IS signed off — but it must never be described as
# "learned" anywhere (docs, logs, UI copy), and any future change to it needs
# the same sign-off, not an engineering judgment call.
#
# A neutral equal-weight prior (the shape to fall back to if no domain-
# approved prior existed) was considered and deliberately not kept as a
# named constant here: it had no runtime, baseline, or evaluation caller —
# nothing in this module chooses between it and DOMAIN_PRIOR_BP at
# resolution time. Reintroducing one when a real evaluation baseline needs
# it is a one-line addition; keeping an unused constant around is not.
#
# OPEN QUESTION — raise with the product owner, not resolved here:
# operator_skill at 35% gives very high authority to a three-tier bucket
# derived from a ten-job rolling window, which most tenants cannot compute in
# year one. Implemented as signed regardless.
DOMAIN_PRIOR_BP: dict[str, int] = {
    "time_overrun": 4000,
    "operator_skill": 3500,
    "seasonality": 1000,
    "material_availability": 1000,
    "supplier_reliability": 500,
}
# The active v1 default. A product-approved domain prior exists (above), so
# this is what ships.
DEFAULT_PRIOR_BP: dict[str, int] = DOMAIN_PRIOR_BP

# Per-signal, absolute — not a flat +/-10%. A flat band is the wrong shape: on
# a 500bp prior +/-1000bp is a 3x move, on a 4000bp prior it's a 25% move.
DEFAULT_BOUNDS_BP: dict[str, Bounds] = {
    "time_overrun": Bounds(3000, 5000),
    "operator_skill": Bounds(2000, 4500),
    "seasonality": Bounds(500, 2000),
    "material_availability": Bounds(500, 2500),
    "supplier_reliability": Bounds(300, 1500),
}

DEFAULT_MAX_PROJECTION_ITERATIONS = 20
DEFAULT_SHRINKAGE_K = 40


@dataclass(frozen=True)
class HistoryPolicyConfig:
    """§6 sufficiency thresholds, extended per Improvement 1 with the hard
    admissibility floor the shrinkage formula gates on. Every number the
    policy reads lives here — no magic numbers in history_policy.py.

    ``events_per_parameter`` answers a DIFFERENT question from
    ``shrinkage_k`` (``WeightAgentConfig.shrinkage_k``), and the two must
    never be derived from one another:

    * ``events_per_parameter`` -> is there enough data for a fitted estimate
      to be statistically usable AT ALL. This scales with the number of free
      weight parameters being estimated (see history_policy.compute_usable_floor)
      — a tenant with fewer available signals has fewer parameters to
      identify and so needs less data to clear the floor.
    * ``shrinkage_k`` -> once admissible, how fast does trust in the fitted
      vector grow with more data (blend.compute_lambda_bp).

    ``min_delayed_work_orders`` (the "sufficient" tier's own, higher,
    fixed threshold) is a separate, admin-facing sufficiency bar, not the
    same thing as the usable floor — see history_policy.assess_history.
    """

    min_history_span_days: int = 180
    preferred_history_span_days: int = 365
    min_completed_work_orders: int = 500
    min_delayed_work_orders: int = 50
    min_signal_coverage: float = 0.80
    # Rule-of-thumb minimum observations per free weight parameter before a
    # fitted estimate is even admissible (a classical "N events per
    # predictor" heuristic — see history_policy.compute_usable_floor for the
    # free-parameter count itself, which depends on how many signals are
    # actually available for a given tenant, not a fixed 5).
    events_per_parameter: int = 10

    def __post_init__(self) -> None:
        for name in (
            "min_history_span_days",
            "preferred_history_span_days",
            "min_completed_work_orders",
            "min_delayed_work_orders",
        ):
            value = getattr(self, name)
            if value < 0:
                raise WeightConfigError(f"{name}={value} must be >= 0")
        if self.events_per_parameter < 1:
            raise WeightConfigError(
                f"events_per_parameter={self.events_per_parameter} must be >= 1"
            )
        if not 0.0 <= self.min_signal_coverage <= 1.0:
            raise WeightConfigError(
                f"min_signal_coverage={self.min_signal_coverage} must be within [0, 1]"
            )
        if self.preferred_history_span_days < self.min_history_span_days:
            raise WeightConfigError(
                "preferred_history_span_days must be >= min_history_span_days"
            )


@dataclass(frozen=True)
class WeightAgentConfig:
    """Self-validating: constructing one directly (not just via
    :func:`build_config`) still enforces every invariant, so nothing
    downstream needs to remember to call a separate validation step."""

    prior_bp: dict[str, int]
    bounds_bp: dict[str, Bounds]
    max_projection_iterations: int
    history_policy: HistoryPolicyConfig
    shrinkage_k: int
    llm_enabled: bool
    max_tenant_description_chars: int = 4000
    # §4.5 lists "profile confidence below threshold" as its own fallback
    # trigger but §3 doesn't name the field — added here rather than as a
    # second, module-local default, since this agent needs one value and
    # this is the single source of truth for every Weight Agent number.
    min_profile_confidence: float = 0.3

    def __post_init__(self) -> None:
        prior_signals = set(self.prior_bp)
        bounds_signals = set(self.bounds_bp)
        if prior_signals != SIGNAL_SET:
            raise WeightConfigError(
                f"prior_bp must have exactly the signals {sorted(SIGNAL_ORDER)}, got "
                f"{sorted(prior_signals)}"
            )
        if bounds_signals != SIGNAL_SET:
            raise WeightConfigError(
                f"bounds_bp must have exactly the signals {sorted(SIGNAL_ORDER)}, got "
                f"{sorted(bounds_signals)}"
            )

        prior_sum = sum(self.prior_bp.values())
        if prior_sum != TOTAL_BP:
            raise WeightConfigError(f"prior_bp must sum to {TOTAL_BP}, got {prior_sum}")

        for signal in SIGNAL_ORDER:
            bound = self.bounds_bp[signal]
            prior = self.prior_bp[signal]
            if not bound.contains(prior):
                raise WeightConfigError(
                    f"prior_bp[{signal!r}]={prior} is outside its bounds "
                    f"[{bound.min}, {bound.max}]"
                )

        sum_min = sum(self.bounds_bp[s].min for s in SIGNAL_ORDER)
        sum_max = sum(self.bounds_bp[s].max for s in SIGNAL_ORDER)
        if not sum_min <= TOTAL_BP <= sum_max:
            raise WeightConfigError(
                f"bounds_bp is infeasible: sum(min)={sum_min}, sum(max)={sum_max}, "
                f"neither brackets {TOTAL_BP} — no weight vector can satisfy every "
                "bound and sum to exactly 10000"
            )

        if self.max_projection_iterations < 1:
            raise WeightConfigError("max_projection_iterations must be >= 1")
        if self.shrinkage_k < 0:
            raise WeightConfigError("shrinkage_k must be >= 0")
        if self.max_tenant_description_chars < 1:
            raise WeightConfigError("max_tenant_description_chars must be >= 1")
        if not 0.0 <= self.min_profile_confidence <= 1.0:
            raise WeightConfigError("min_profile_confidence must be within [0, 1]")


def build_config(
    *,
    prior_bp: dict[str, int] | None = None,
    bounds_bp: dict[str, Bounds] | None = None,
    max_projection_iterations: int = DEFAULT_MAX_PROJECTION_ITERATIONS,
    history_policy: HistoryPolicyConfig | None = None,
    shrinkage_k: int = DEFAULT_SHRINKAGE_K,
    llm_enabled: bool | None = None,
    max_tenant_description_chars: int = 4000,
    min_profile_confidence: float = 0.3,
) -> WeightAgentConfig:
    """Construct a config. Defaults reproduce the shipped v1 values; tests
    pass overrides to exercise infeasible configs deliberately. Validation
    itself lives on ``WeightAgentConfig.__post_init__`` — this function only
    supplies defaults and the ``Settings`` lookup for ``llm_enabled``.

    ``shrinkage_k`` and ``history_policy.events_per_parameter`` are
    deliberately independent — passing a non-default ``shrinkage_k`` here
    does NOT change the usable floor; see ``HistoryPolicyConfig``'s
    docstring for why the two must never be derived from one another.
    """
    if llm_enabled is None:
        llm_enabled = get_settings().m3_weight_llm_enabled
    if history_policy is None:
        history_policy = HistoryPolicyConfig()
    return WeightAgentConfig(
        prior_bp=dict(prior_bp if prior_bp is not None else DEFAULT_PRIOR_BP),
        bounds_bp=dict(bounds_bp if bounds_bp is not None else DEFAULT_BOUNDS_BP),
        max_projection_iterations=max_projection_iterations,
        history_policy=history_policy,
        shrinkage_k=shrinkage_k,
        llm_enabled=llm_enabled,
        max_tenant_description_chars=max_tenant_description_chars,
        min_profile_confidence=min_profile_confidence,
    )


@functools.lru_cache(maxsize=1)
def get_weight_agent_config() -> WeightAgentConfig:
    """Cached singleton — the shipped v1 config. Call
    ``get_weight_agent_config.cache_clear()`` in tests to reload."""
    return build_config()
