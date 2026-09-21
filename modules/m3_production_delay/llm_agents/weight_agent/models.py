"""Domain types for the Weight Agent — the resolution output contract (spec §9)
plus the value objects every stage (config, availability, blend, projection,
LLM adjustment) passes between them.

Weights are integer basis points everywhere in this module. A float only ever
appears at the public boundary (:func:`bp_to_percent`), explicitly derived and
lossy — never stored, compared, or summed.

Every dataclass here validates itself in ``__post_init__``. Nothing downstream
(the resolver included) should have to remember to re-check an invariant a
constructor could have enforced — an invalid domain object should simply not
be constructible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from m3_production_delay.llm_agents.weight_agent.exceptions import FittedWeightsError

# Canonical signal order (spec §1). Normative for every deterministic
# iteration and tie-break — projection, redistribution and blend rounding all
# walk signals in exactly this order so results are reproducible.
SIGNAL_ORDER: tuple[str, ...] = (
    "time_overrun",
    "operator_skill",
    "seasonality",
    "material_availability",
    "supplier_reliability",
)
SIGNAL_SET: frozenset[str] = frozenset(SIGNAL_ORDER)

# A Literal alias for the same five values. This is what lets a type checker
# catch a typo'd signal name (e.g. "operater_skill") at analysis time instead
# of only at __post_init__ runtime — the frozenset check stays too, since a
# value arriving from JSON/LLM output is unchecked by the type system either
# way.
SignalName = Literal[
    "time_overrun",
    "operator_skill",
    "seasonality",
    "material_availability",
    "supplier_reliability",
]

TOTAL_BP = 10_000

# Closed enums (spec §9). ``Literal`` gives static typos ("configureed") a
# type-checker error; the frozensets below give the same guarantee at runtime
# for values arriving from outside the type system (JSON, LLM output).
SOURCE_CONFIGURED = "configured"
SOURCE_HISTORICALLY_FITTED = "historically_fitted"
SOURCE_BLENDED = "blended"
SOURCE_LLM_ADJUSTED_PRIOR = "llm_adjusted_prior"
SOURCE_PRIOR = "prior"
SourceName = Literal[
    "configured", "historically_fitted", "blended", "llm_adjusted_prior", "prior"
]
VALID_SOURCES: frozenset[str] = frozenset(
    {
        SOURCE_CONFIGURED,
        SOURCE_HISTORICALLY_FITTED,
        SOURCE_BLENDED,
        SOURCE_LLM_ADJUSTED_PRIOR,
        SOURCE_PRIOR,
    }
)

STATUS_ACTIVE = "active"
STATUS_RECOMMENDATION = "recommendation"
StatusName = Literal["active", "recommendation"]
VALID_STATUSES: frozenset[str] = frozenset({STATUS_ACTIVE, STATUS_RECOMMENDATION})

# §6/Improvement 1 history admissibility tiers — distinct from the single
# sufficient/insufficient boolean the original spec's output contract uses.
# "inadmissible" and "usable_weak" both report sufficient=False (nothing
# below the "sufficient" tier is admin-facing-trustworthy) but only
# "inadmissible" forces lambda_bp to 0 outright; "usable_weak" still gets a
# small, continuously-scaled fitted contribution. See blend.compute_lambda_bp.
HistoryAdmissibility = Literal["inadmissible", "usable_weak", "sufficient", "preferred"]
ADMISSIBILITY_INADMISSIBLE: HistoryAdmissibility = "inadmissible"
ADMISSIBILITY_USABLE_WEAK: HistoryAdmissibility = "usable_weak"
ADMISSIBILITY_SUFFICIENT: HistoryAdmissibility = "sufficient"
ADMISSIBILITY_PREFERRED: HistoryAdmissibility = "preferred"
VALID_ADMISSIBILITY: frozenset[str] = frozenset(
    {
        ADMISSIBILITY_INADMISSIBLE,
        ADMISSIBILITY_USABLE_WEAK,
        ADMISSIBILITY_SUFFICIENT,
        ADMISSIBILITY_PREFERRED,
    }
)

# --- §7 tenant profile vocabulary -------------------------------------------
# Lives here (not in the LLM-calling code) so TenantProfile can validate
# itself against it in __post_init__ regardless of which caller constructs
# one. v1 placeholder taxonomy — needs product/domain sign-off, same spirit
# as DOMAIN_PRIOR_BP in config.py.
PROFILE_FIELDS: dict[str, frozenset[str]] = {
    "industry": frozenset(
        {"manufacturing_discrete", "manufacturing_process", "fabrication", "assembly", "other"}
    ),
    "production_type": frozenset(
        {"make_to_order", "make_to_stock", "engineer_to_order", "hybrid"}
    ),
    "material_dependency": frozenset({"low", "medium", "high"}),
    "supplier_dependency": frozenset({"low", "medium", "high"}),
    "workforce_dependency": frozenset({"low", "medium", "high"}),
    "workforce_stability": frozenset({"stable", "moderate", "volatile"}),
    "seasonality_level": frozenset({"none", "low", "moderate", "high"}),
    "automation_level": frozenset({"manual", "semi_automated", "highly_automated"}),
    "make_to_order_ratio": frozenset({"low", "medium", "high"}),
    "supply_chain_complexity": frozenset({"simple", "moderate", "complex"}),
}

# Fields whose absence makes a profile too thin to drive a numeric adjustment
# from (Improvement 2). Reviewed set: the four dependency/exposure axes plus
# production_type — the fields the adjustment prompt actually reasons about
# per-signal (time_overrun/operator_skill are cross-cutting and don't map to
# one profile field, so they aren't gated on any single field here).
CRITICAL_PROFILE_FIELDS: frozenset[str] = frozenset(
    {
        "production_type",
        "material_dependency",
        "supplier_dependency",
        "workforce_dependency",
        "seasonality_level",
    }
)


def bp_to_percent(weights_bp: dict[str, int]) -> dict[str, float]:
    """Public-boundary float view of a basis-point map: ``percent = bp / 100``.

    Derived and lossy — never feed this back into internal arithmetic.
    """
    return {signal: bp / 100 for signal, bp in weights_bp.items()}


@dataclass(frozen=True)
class Bounds:
    """Per-signal ``[min, max]`` in basis points, inclusive. Self-validating:
    an inverted or out-of-range Bounds cannot be constructed, so nothing
    downstream needs to re-check ``min <= max`` before using one."""

    min: int
    max: int

    def __post_init__(self) -> None:
        if not (0 <= self.min <= self.max <= TOTAL_BP):
            raise ValueError(
                f"Bounds(min={self.min}, max={self.max}) must satisfy "
                f"0 <= min <= max <= {TOTAL_BP}"
            )

    def clip(self, value: int) -> int:
        return max(self.min, min(self.max, value))

    def contains(self, value: int) -> bool:
        return self.min <= value <= self.max


@dataclass(frozen=True)
class ExcludedSignal:
    signal: str
    reason: str

    def __post_init__(self) -> None:
        if self.signal not in SIGNAL_SET:
            raise ValueError(f"unknown signal {self.signal!r}")
        if not self.reason:
            raise ValueError("reason must not be empty")


@dataclass(frozen=True)
class HistoryAssessment:
    """§6 sufficiency verdict, extended with the §1-improvement admissibility
    tier. ``reasons`` lists only the FAILING criteria for the "sufficient"
    bundle — empty when sufficient. ``sufficient`` is a derived convenience
    (``admissibility in {sufficient, preferred}``); ``admissibility`` is the
    finer-grained classification of the CURRENT TENANT's resolution-time
    history posture — audit/UI-facing, computed from ``HistoryInputs``.

    ``lambda_bp`` is the shrinkage weight actually applied to the fitted
    vector in the blend. As of the fitted-artifact-provenance design, this is
    NOT derived from the same inputs as ``admissibility``: it comes from the
    fitted artifact's OWN declared sample count (``FittedWeights.
    delayed_event_count``), so that an old, well-evidenced fit is not made to
    look weaker by a tenant's currently-thin history, and a thin, recent fit
    is not made to look stronger by a tenant's currently-rich history (§5).
    ``admissibility=inadmissible`` together with ``lambda_bp>0`` is therefore
    a legitimate, expected combination — it says "this tenant's own recent
    history doesn't clear the bar, but the fitted artifact backing this
    recommendation carries its own sufficient evidence" — not a bug.
    """

    admissibility: HistoryAdmissibility
    reasons: tuple[str, ...] = ()
    lambda_bp: int = 0

    def __post_init__(self) -> None:
        if self.admissibility not in VALID_ADMISSIBILITY:
            raise ValueError(f"unknown admissibility {self.admissibility!r}")
        if not 0 <= self.lambda_bp <= TOTAL_BP:
            raise ValueError(f"lambda_bp={self.lambda_bp} out of [0, {TOTAL_BP}]")

    @property
    def sufficient(self) -> bool:
        return self.admissibility in (ADMISSIBILITY_SUFFICIENT, ADMISSIBILITY_PREFERRED)


@dataclass(frozen=True)
class HistoryInputs:
    """Caller-supplied observations the §6 policy and §4.3 blend read.

    The agent does not compute these itself — they come from wherever the
    fitting pipeline (not built here) or its caller already tracks them.
    """

    history_span_days: int
    completed_work_orders: int
    delayed_work_orders: int
    per_signal_coverage: dict[str, float] = field(default_factory=dict)  # signal -> 0..1

    def __post_init__(self) -> None:
        for field_name, value in (
            ("history_span_days", self.history_span_days),
            ("completed_work_orders", self.completed_work_orders),
            ("delayed_work_orders", self.delayed_work_orders),
        ):
            if value < 0:
                raise ValueError(f"{field_name}={value} must be >= 0")
        if self.delayed_work_orders > self.completed_work_orders:
            raise ValueError(
                f"delayed_work_orders={self.delayed_work_orders} cannot exceed "
                f"completed_work_orders={self.completed_work_orders}"
            )
        for signal, coverage in self.per_signal_coverage.items():
            if signal not in SIGNAL_SET:
                raise ValueError(f"unknown signal {signal!r} in per_signal_coverage")
            if not (math.isfinite(coverage) and 0.0 <= coverage <= 1.0):
                raise ValueError(f"per_signal_coverage[{signal!r}]={coverage} must be in [0, 1]")


# Reason codes FittedWeights.__post_init__ raises with, as a message prefix
# (not a separate field) — greppable/testable via substring, same pattern as
# history_policy.py's reason constants, without adding a second field to an
# already-simple exception.
FITTED_WEIGHTS_WRONG_SIGNAL_SET = "fitted_weights_wrong_signal_set"
FITTED_WEIGHTS_INCONSISTENT_WITH_SIGNAL_SET = "fitted_weights_inconsistent_with_signal_set"
FITTED_WEIGHTS_BAD_SUM = "fitted_weights_bad_sum"


@dataclass(frozen=True)
class FittedWeights:
    """The output of a (not-yet-built) historical fitting run, as it would
    arrive through :class:`FittedWeightsProvider`. Self-validates the same
    structural invariants every other source is held to (Improvement 8), plus
    the provenance a fitted artifact must declare to be usable at all: which
    signals it was actually fitted over, and how much data justified it.

    Contract (deliberately the one described, not left ambiguous):
    ``weights_bp`` is always the complete five-signal mapping — signals
    outside ``signal_set`` MUST be exactly 0, and the signals INSIDE
    ``signal_set`` must sum to exactly 10000. This makes the artifact
    self-contained for serialization/auditing while keeping fit provenance
    explicit: a signal being 0 because it was never fitted (excluded from
    ``signal_set``) is a different fact from a signal being 0 because the fit
    assigned it no weight (included in ``signal_set`` with value 0), and only
    the dict alone can't tell those apart.

    A fitted vector is NOT automatically valid for a different signal set
    than the one it was fitted over — see resolver.py's compatibility check,
    which compares ``signal_set`` against current availability by exact set
    equality, not count, before this artifact is ever blended.
    """

    weights_bp: dict[str, int]
    signal_set: frozenset[SignalName]
    delayed_event_count: int

    def __post_init__(self) -> None:
        if not self.signal_set:
            raise FittedWeightsError(
                f"{FITTED_WEIGHTS_WRONG_SIGNAL_SET}: signal_set must not be empty"
            )
        unknown = self.signal_set - SIGNAL_SET
        if unknown:
            raise FittedWeightsError(
                f"{FITTED_WEIGHTS_WRONG_SIGNAL_SET}: signal_set contains unknown "
                f"signals {sorted(unknown)}"
            )
        # frozenset already makes duplicate membership impossible to
        # construct — no separate check needed for "no duplicates".

        keys = set(self.weights_bp)
        if keys != SIGNAL_SET:
            raise FittedWeightsError(
                f"{FITTED_WEIGHTS_WRONG_SIGNAL_SET}: weights_bp must have exactly the "
                f"signals {sorted(SIGNAL_ORDER)}, got {sorted(keys)}"
            )
        for signal, value in self.weights_bp.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise FittedWeightsError(f"FittedWeights[{signal!r}]={value!r} is not an integer")
            if value < 0:
                raise FittedWeightsError(f"FittedWeights[{signal!r}]={value} must be >= 0")

        outside = SIGNAL_SET - self.signal_set
        nonzero_outside = {signal: self.weights_bp[signal] for signal in outside if self.weights_bp[signal] != 0}
        if nonzero_outside:
            raise FittedWeightsError(
                f"{FITTED_WEIGHTS_INCONSISTENT_WITH_SIGNAL_SET}: nonzero weight(s) "
                f"outside signal_set: {sorted(nonzero_outside)}"
            )

        fitted_total = sum(self.weights_bp[signal] for signal in self.signal_set)
        if fitted_total != TOTAL_BP:
            raise FittedWeightsError(
                f"{FITTED_WEIGHTS_BAD_SUM}: weights across signal_set must sum to "
                f"{TOTAL_BP}, got {fitted_total}"
            )

        if self.delayed_event_count < 0:
            raise FittedWeightsError(
                f"delayed_event_count={self.delayed_event_count} must be >= 0"
            )


@dataclass(frozen=True)
class TenantProfile:
    """Structured extraction of tenant prose (spec §7). Every field is either
    a value from that field's closed vocabulary (:data:`PROFILE_FIELDS`) or
    ``None``. Self-validating: an out-of-enum value cannot be smuggled into a
    constructed profile, so nothing downstream needs to re-check it."""

    industry: str | None = None
    production_type: str | None = None
    material_dependency: str | None = None
    supplier_dependency: str | None = None
    workforce_dependency: str | None = None
    workforce_stability: str | None = None
    seasonality_level: str | None = None
    automation_level: str | None = None
    make_to_order_ratio: str | None = None
    supply_chain_complexity: str | None = None

    def __post_init__(self) -> None:
        for name, allowed in PROFILE_FIELDS.items():
            value = getattr(self, name)
            if value is not None and value not in allowed:
                raise ValueError(f"TenantProfile.{name}={value!r} is not in {sorted(allowed)}")

    def fields(self) -> dict[str, str | None]:
        return {name: getattr(self, name) for name in PROFILE_FIELDS}

    def missing_critical_fields(self) -> tuple[str, ...]:
        # Iterates PROFILE_FIELDS (insertion-ordered) rather than the
        # frozenset CRITICAL_PROFILE_FIELDS directly, so the result is
        # deterministic — consistent with every other ordering guarantee in
        # this package (see SIGNAL_ORDER's docstring).
        return tuple(
            name
            for name in PROFILE_FIELDS
            if name in CRITICAL_PROFILE_FIELDS and getattr(self, name) is None
        )


@dataclass(frozen=True)
class WeightResolution:
    """The §9 output contract. Trusted fields (tenant_id, source, status,
    requires_admin_approval, confidence, generated_at, prompt_version,
    transform_version, agent_version, bounds_bp, prior_bp) are set only by
    :mod:`resolver` — never copied from an LLM response. Self-validates the
    numeric invariants every source must satisfy, not just the source/status
    pairing — a resolver bug that skipped a check fails loudly here instead
    of shipping a malformed recommendation.
    """

    tenant_id: str
    source: str
    status: str
    weights_bp: dict[str, int]
    available_signals: tuple[str, ...]
    excluded_signals: tuple[ExcludedSignal, ...]
    history_assessment: HistoryAssessment
    prior_bp: dict[str, int]
    adjustments_bp: dict[str, int]
    bounds_bp: dict[str, Bounds]
    confidence: float
    evidence: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    requires_admin_approval: bool
    prompt_version: str
    transform_version: str
    agent_version: str
    generated_at: str
    # Audit-only: was fitted.weights_bp projected into the current tenant's
    # Bounds before blending (fitted was structurally valid and signal-set
    # compatible, but violated a per-signal bound)? Only ever true for
    # source in {blended, historically_fitted} — every other source leaves
    # these at their defaults, since fitted projection doesn't apply there.
    fitted_projection_applied: bool = False
    fitted_projection_changed_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id must not be empty")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"unknown status {self.status!r}")
        if self.status == STATUS_ACTIVE and self.source != SOURCE_CONFIGURED:
            raise ValueError("status=active is only valid for source=configured")
        if self.requires_admin_approval and self.source == SOURCE_CONFIGURED:
            raise ValueError("source=configured must not require admin approval")
        if not self.requires_admin_approval and self.source != SOURCE_CONFIGURED:
            raise ValueError(f"source={self.source!r} must require admin approval")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence={self.confidence} must be a finite value in [0, 1]")
        if not self.available_signals:
            raise ValueError(
                "WeightResolution cannot be constructed with zero available signals — "
                "resolver.py must raise AllSignalsUnavailableError before reaching this "
                "point, never build a degenerate all-zero result"
            )
        if self.fitted_projection_applied and self.source not in (SOURCE_BLENDED, SOURCE_HISTORICALLY_FITTED):
            raise ValueError(
                f"fitted_projection_applied=True is only meaningful for source in "
                f"{{{SOURCE_BLENDED!r}, {SOURCE_HISTORICALLY_FITTED!r}}}, got {self.source!r}"
            )
        if not self.fitted_projection_applied and self.fitted_projection_changed_signals:
            raise ValueError(
                "fitted_projection_changed_signals must be empty when "
                "fitted_projection_applied is False"
            )
        unknown_changed = set(self.fitted_projection_changed_signals) - SIGNAL_SET
        if unknown_changed:
            raise ValueError(
                f"fitted_projection_changed_signals contains unknown signals {sorted(unknown_changed)}"
            )

        # sum(weights_bp) == 10000 is unconditional (spec §2). There is no
        # degenerate "all zero" exception here: a WeightResolution with
        # available_signals=() cannot exist at all — resolver.py raises
        # AllSignalsUnavailableError before ever constructing one, precisely
        # because zero available signals means there is no valid vector to
        # represent, not a vector that happens to be all zeros.
        for dict_name, values, expect_sum in (
            ("weights_bp", self.weights_bp, TOTAL_BP),
            ("prior_bp", self.prior_bp, TOTAL_BP),
            ("adjustments_bp", self.adjustments_bp, 0),
        ):
            keys = set(values)
            if keys != SIGNAL_SET:
                raise ValueError(
                    f"{dict_name} must have exactly the signals {sorted(SIGNAL_ORDER)}, "
                    f"got {sorted(keys)}"
                )
            for signal, value in values.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{dict_name}[{signal!r}]={value!r} is not an integer")
            if expect_sum is not None and sum(values.values()) != expect_sum:
                raise ValueError(
                    f"{dict_name} must sum to {expect_sum}, got {sum(values.values())}"
                )
        if set(self.bounds_bp) != SIGNAL_SET:
            raise ValueError(
                f"bounds_bp must have exactly the signals {sorted(SIGNAL_ORDER)}, "
                f"got {sorted(self.bounds_bp)}"
            )
        for signal in SIGNAL_ORDER:
            if not self.bounds_bp[signal].contains(self.weights_bp[signal]):
                raise ValueError(
                    f"weights_bp[{signal!r}]={self.weights_bp[signal]} is outside "
                    f"bounds_bp[{signal!r}]={self.bounds_bp[signal]}"
                )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "source": self.source,
            "status": self.status,
            "weights_bp": dict(self.weights_bp),
            "available_signals": list(self.available_signals),
            "excluded_signals": [
                {"signal": e.signal, "reason": e.reason} for e in self.excluded_signals
            ],
            "history_assessment": {
                "sufficient": self.history_assessment.sufficient,
                "admissibility": self.history_assessment.admissibility,
                "reasons": list(self.history_assessment.reasons),
                "lambda_bp": self.history_assessment.lambda_bp,
            },
            "prior_bp": dict(self.prior_bp),
            "adjustments_bp": dict(self.adjustments_bp),
            "bounds_bp": {
                signal: {"min": b.min, "max": b.max} for signal, b in self.bounds_bp.items()
            },
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "fallback_reasons": list(self.fallback_reasons),
            "requires_admin_approval": self.requires_admin_approval,
            "prompt_version": self.prompt_version,
            "transform_version": self.transform_version,
            "agent_version": self.agent_version,
            "generated_at": self.generated_at,
            "fitted_projection_applied": self.fitted_projection_applied,
            "fitted_projection_changed_signals": list(self.fitted_projection_changed_signals),
        }
