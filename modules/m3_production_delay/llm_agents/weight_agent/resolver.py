"""§4 resolution flow: one path, four sources, evaluated in priority order.
This is the clean interface the M3 orchestrator (``occustrator.py``) calls —
``WeightAgent().resolve(...)`` — everything else in this package is a detail
this class wires together.

No per-tenant mutable state is held anywhere in this class: every dependency
is either stateless (the Null provider) or passed in per call. §10 caching
decision for v1: **no caching at all** — the simplest option, and it
trivially satisfies "a recommendation for tenant A is never served to tenant
B" because nothing is ever retained between calls.
"""

from __future__ import annotations

from maxxflow_core.clock import get_clock
from maxxflow_core.errors import get_logger
from maxxflow_core.ports import LLMProvider
from maxxflow_providers import get_llm_provider

from m3_production_delay.llm_agents.weight_agent.availability import apply_availability
from m3_production_delay.llm_agents.weight_agent.blend import blend_weights, compute_lambda_bp
from m3_production_delay.llm_agents.weight_agent.config import WeightAgentConfig, get_weight_agent_config
from m3_production_delay.llm_agents.weight_agent.exceptions import (
    AllSignalsUnavailableError,
    FittedWeightsError,
    LLMAdjustmentError,
    WeightConfigError,
)
from m3_production_delay.llm_agents.weight_agent.history_policy import assess_history, compute_usable_floor
from m3_production_delay.llm_agents.weight_agent.models import (
    ADMISSIBILITY_INADMISSIBLE,
    SIGNAL_ORDER,
    SIGNAL_SET,
    SOURCE_BLENDED,
    SOURCE_CONFIGURED,
    SOURCE_HISTORICALLY_FITTED,
    SOURCE_LLM_ADJUSTED_PRIOR,
    SOURCE_PRIOR,
    STATUS_ACTIVE,
    STATUS_RECOMMENDATION,
    TOTAL_BP,
    Bounds,
    HistoryAssessment,
    HistoryInputs,
    WeightResolution,
)
from m3_production_delay.llm_agents.weight_agent.profile_extractor import (
    PROMPT_VERSION as PROFILE_PROMPT_VERSION,
    TenantProfileExtractor,
)
from m3_production_delay.llm_agents.weight_agent.projection import project_adjustment
from m3_production_delay.llm_agents.weight_agent.providers import (
    FittedWeightsProvider,
    NullFittedWeightsProvider,
)
from m3_production_delay.llm_agents.weight_agent.tracing import NOOP_TRACER, WeightAgentTracer
from m3_production_delay.llm_agents.weight_agent.weight_adjustment import (
    PROMPT_VERSION as ADJUSTMENT_PROMPT_VERSION,
    WeightAdjustmentGenerator,
)

log = get_logger("m3_production_delay.weight_agent.resolver")

AGENT_VERSION = "v1"
TRANSFORM_VERSION = "v1"
# Two prompts now exist (Improvement 2) — the output contract has one
# prompt_version field, so both versions are reported together rather than
# picking one and losing the other's identity for reproduction/diffing.
COMBINED_PROMPT_VERSION = f"profile={PROFILE_PROMPT_VERSION};adjustment={ADJUSTMENT_PROMPT_VERSION}"

# Fallback reasons for the fitted-weight boundary (§3/§8). Named constants,
# not free text scattered inline, so a test (or a future log query) can match
# on the exact code rather than a substring of a hand-written sentence.
FITTED_WEIGHTS_INVALID = "fitted_weights_invalid"
FITTED_PROVIDER_ERROR = "fitted_provider_error"
FITTED_SIGNAL_SET_MISMATCH = "fitted_signal_set_mismatch"
FITTED_HISTORY_BELOW_USABLE_FLOOR = "fitted_history_below_usable_floor"
FITTED_PROJECTION_INFEASIBLE = "fitted_projection_infeasible"

# Reused by both the pre-blend fitted-vector projection and the post-blend
# final projection — projecting a vector into Bounds is exactly
# project_adjustment with a zero adjustment (clip, then redistribute the
# residual). No second projection algorithm.
_ZERO_ADJUSTMENT = {signal: 0 for signal in SIGNAL_ORDER}

_NOT_APPLICABLE_HISTORY = HistoryAssessment(
    admissibility=ADMISSIBILITY_INADMISSIBLE,
    reasons=("not_applicable_source_configured",),
    lambda_bp=0,
)


def _validate_configured_weights(weights_bp: dict[str, int], bounds_bp: dict[str, Bounds]) -> None:
    keys = set(weights_bp)
    if keys != SIGNAL_SET:
        raise WeightConfigError(
            f"configured weights must have exactly the signals {sorted(SIGNAL_ORDER)}, "
            f"got {sorted(keys)}"
        )
    total = sum(weights_bp.values())
    if total != TOTAL_BP:
        raise WeightConfigError(f"configured weights must sum to {TOTAL_BP}, got {total}")
    for signal in SIGNAL_ORDER:
        bound = bounds_bp[signal]
        value = weights_bp[signal]
        if not bound.contains(value):
            raise WeightConfigError(
                f"configured weights[{signal!r}]={value} is outside its bounds "
                f"[{bound.min}, {bound.max}]"
            )


class WeightAgent:
    def __init__(
        self,
        *,
        fitted_provider: FittedWeightsProvider | None = None,
        llm_provider: LLMProvider | None = None,
        config: WeightAgentConfig | None = None,
    ) -> None:
        self._fitted_provider = fitted_provider or NullFittedWeightsProvider()
        self._llm_provider = llm_provider
        self._config = config

    def _resolve_config(self) -> WeightAgentConfig:
        return self._config if self._config is not None else get_weight_agent_config()

    def _resolve_llm_provider(self) -> LLMProvider:
        return self._llm_provider if self._llm_provider is not None else get_llm_provider()

    def resolve(
        self,
        tenant_id: str,
        *,
        availability: dict[str, bool],
        configured_bp: dict[str, int] | None = None,
        exclusion_reasons: dict[str, str] | None = None,
        tenant_description: str | None = None,
        history_inputs: HistoryInputs | None = None,
        tracer: WeightAgentTracer = NOOP_TRACER,
    ) -> WeightResolution:
        config = self._resolve_config()
        generated_at = get_clock().now_utc().isoformat()
        exclusion_reasons = exclusion_reasons or {}
        fallback_reasons: list[str] = []

        # --- 4.1 configured weights (highest priority) ----------------------
        if configured_bp is not None:
            try:
                _validate_configured_weights(configured_bp, config.bounds_bp)
                tracer.trace_stage(
                    "WEIGHT RESOLUTION", route="configured", configured_weights=True, llm_skipped=True
                )
                result = WeightResolution(
                    tenant_id=tenant_id,
                    source=SOURCE_CONFIGURED,
                    status=STATUS_ACTIVE,
                    weights_bp=dict(configured_bp),
                    available_signals=SIGNAL_ORDER,
                    excluded_signals=(),
                    history_assessment=_NOT_APPLICABLE_HISTORY,
                    prior_bp=dict(config.prior_bp),
                    adjustments_bp={signal: 0 for signal in SIGNAL_ORDER},
                    bounds_bp=dict(config.bounds_bp),
                    confidence=1.0,
                    evidence=(),
                    fallback_reasons=(),
                    requires_admin_approval=False,
                    prompt_version=COMBINED_PROMPT_VERSION,
                    transform_version=TRANSFORM_VERSION,
                    agent_version=AGENT_VERSION,
                    generated_at=generated_at,
                )
                self._audit_log(
                    result, fitted_available=False, llm_profile_used=False, llm_adjustment_used=False
                )
                tracer.trace_final_summary(result)
                return result
            except WeightConfigError as exc:
                # Validation failure on stored config is an error condition,
                # not a silent fallback — log it, then fall through to the
                # prior with an explicit reason recorded.
                log.error("tenant %s has invalid configured weights: %s", tenant_id, exc)
                fallback_reasons.append(f"configured_weights_invalid: {exc}")
                tracer.trace_decision("WEIGHT RESOLUTION", "configured_invalid", error=str(exc))

        # --- 4.2 signal availability mask (every non-configured path) ------
        availability_result = apply_availability(
            config.prior_bp, config.bounds_bp, availability, SIGNAL_ORDER, exclusion_reasons
        )
        if not availability_result.available_signals:
            # No valid weight vector exists with zero available signals —
            # this must not be represented as a resolved (if degenerate)
            # recommendation. Raised, not swallowed into fallback_reasons,
            # because unlike an LLM or fitted-provider failure there is no
            # safe substitute to fall back to.
            log.error(
                "tenant %s: all five signals unavailable, no weight vector is possible",
                tenant_id,
            )
            tracer.trace_decision(
                "WEIGHT RESOLUTION",
                "all_signals_unavailable",
                configured_weights=False,
                excluded_signals=[e.signal for e in availability_result.excluded_signals],
            )
            raise AllSignalsUnavailableError(
                f"tenant {tenant_id}: every signal is unavailable "
                f"({[e.reason for e in availability_result.excluded_signals]})"
            )
        prior_bp = availability_result.prior_bp
        bounds_bp = availability_result.bounds_bp
        fallback_reasons.extend(availability_result.reasons)
        tracer.trace_stage(
            "WEIGHT RESOLUTION",
            configured_weights=False,
            available_signals=list(availability_result.available_signals),
            excluded_signals=[e.signal for e in availability_result.excluded_signals],
        )
        tracer.trace_weights(
            "PRIOR", source="domain_prior", weights_bp=config.prior_bp, effective_prior_bp=prior_bp
        )

        # --- 4.3 historical blend, gated by history admissibility ----------
        # Signal-aware: the usable floor scales with how many signals this
        # tenant actually has available, never a fixed assumption of 5.
        available_count = len(availability_result.available_signals)
        if history_inputs is not None:
            history_assessment = assess_history(history_inputs, config.history_policy, available_count)
        else:
            history_assessment = HistoryAssessment(
                admissibility=ADMISSIBILITY_INADMISSIBLE,
                reasons=("no_history_inputs_supplied",),
                lambda_bp=0,
            )

        # Fitted-weight boundary (§3/§5/§6): fetch -> structural validation
        # (FittedWeights.__post_init__, inside .get()) -> signal-set
        # compatibility -> the fit's OWN sample size for admissibility ->
        # blend only if all of that holds. A malformed or incompatible
        # fitted artifact is never repaired or blended around — it is
        # dropped, with a specific reason recorded, and the resolution
        # continues exactly as if fitted were absent.
        fitted = None
        try:
            fitted = self._fitted_provider.get(tenant_id)
        except FittedWeightsError as exc:
            log.error("tenant %s fitted weights provider returned invalid data: %s", tenant_id, exc)
            fallback_reasons.append(f"{FITTED_WEIGHTS_INVALID}: {exc}")
        except Exception as exc:  # provider transport/runtime failure
            log.error("tenant %s fitted weights provider raised an error: %s", tenant_id, exc)
            fallback_reasons.append(f"{FITTED_PROVIDER_ERROR}: {exc}")

        # Captured before any rejection below, so the audit log distinguishes
        # "the provider returned nothing" from "it returned something that
        # was rejected" (the latter is always visible in fallback_reasons
        # too, via its specific reason code).
        fitted_provider_returned_data = fitted is not None

        if fitted is not None:
            tracer.trace_stage(
                "FITTED ARTIFACT",
                signal_set=sorted(fitted.signal_set),
                delayed_event_count=fitted.delayed_event_count,
            )
            current_signal_set = frozenset(availability_result.available_signals)
            if fitted.signal_set != current_signal_set:
                # Exact set equality, not count: a fit over {time, operator,
                # seasonality, material} is NOT valid for {time, operator,
                # material, supplier} even though both have four signals —
                # the learned coefficients are specific to the predictor set
                # they were fitted against. Never renormalise around this.
                log.warning(
                    "tenant %s fitted signal_set %s does not match current availability %s",
                    tenant_id,
                    sorted(fitted.signal_set),
                    sorted(current_signal_set),
                )
                tracer.trace_decision(
                    "FITTED ARTIFACT",
                    "reject_fitted",
                    reason=FITTED_SIGNAL_SET_MISMATCH,
                    current_available_signal_set=sorted(current_signal_set),
                    signal_set_match=False,
                )
                fallback_reasons.append(FITTED_SIGNAL_SET_MISMATCH)
                fitted = None
            else:
                tracer.trace_stage(
                    "FITTED ARTIFACT",
                    current_available_signal_set=sorted(current_signal_set),
                    signal_set_match=True,
                )

        lambda_bp = 0
        if fitted is not None:
            # The fit's OWN evidence, not resolution-time tenant history —
            # an old fit stays exactly as (in)admissible as when it was
            # produced, regardless of how much history the tenant has
            # accumulated since (§5). history_assessment above still reports
            # the tenant's current posture for audit/UI; it does not gate
            # this blend.
            free_parameters = max(0, len(fitted.signal_set) - 1)
            n_floor = compute_usable_floor(
                len(fitted.signal_set), config.history_policy.events_per_parameter
            )
            lambda_bp = compute_lambda_bp(fitted.delayed_event_count, config.shrinkage_k, n_floor)
            tracer.trace_stage(
                "FITTED ARTIFACT",
                free_parameters=free_parameters,
                events_per_parameter=config.history_policy.events_per_parameter,
                n_floor=n_floor,
                fitted_delayed_events=fitted.delayed_event_count,
                effective_n=max(0, fitted.delayed_event_count - n_floor),
                lambda_bp=lambda_bp,
                fitted_influence=f"{lambda_bp / 100:.2f}%",
                prior_influence=f"{(TOTAL_BP - lambda_bp) / 100:.2f}%",
            )
            if lambda_bp == 0:
                tracer.trace_decision(
                    "FITTED ARTIFACT", "reject_fitted", reason=FITTED_HISTORY_BELOW_USABLE_FLOOR
                )
                fallback_reasons.append(FITTED_HISTORY_BELOW_USABLE_FLOOR)
                fitted = None

        if fitted is not None:
            # A structurally valid, signal-set-compatible, admissible fitted
            # vector can still violate this tenant's configured per-signal
            # Bounds (different concern from signal-set incompatibility —
            # the model is still statistically valid, it just needs to obey
            # a business/product constraint blend_weights itself does not
            # enforce). Project it into bounds_bp BEFORE blending, reusing
            # projection.py rather than blending a vector that doesn't
            # belong to this tenant's admissible weight space.
            tracer.trace_weights(
                "FITTED PROJECTION",
                raw_fitted_bp=fitted.weights_bp,
                bounds_bp={s: f"[{bounds_bp[s].min},{bounds_bp[s].max}]" for s in SIGNAL_ORDER},
            )
            fitted_projection = project_adjustment(
                fitted.weights_bp, bounds_bp, _ZERO_ADJUSTMENT, SIGNAL_ORDER,
                max_iterations=config.max_projection_iterations,
            )
            if fitted_projection.success and fitted_projection.weights_bp is not None:
                projected_fitted_bp = fitted_projection.weights_bp
                fitted_projection_applied = projected_fitted_bp != fitted.weights_bp
                fitted_projection_changed_signals = (
                    tuple(
                        signal for signal in SIGNAL_ORDER
                        if projected_fitted_bp[signal] != fitted.weights_bp[signal]
                    )
                    if fitted_projection_applied
                    else ()
                )
                tracer.trace_stage(
                    "FITTED PROJECTION",
                    projected_fitted_bp=projected_fitted_bp,
                    projection_applied=fitted_projection_applied,
                    changed_signals=list(fitted_projection_changed_signals),
                )

                blended_bp = blend_weights(prior_bp, projected_fitted_bp, lambda_bp, SIGNAL_ORDER)
                tracer.trace_weights(
                    "BLEND",
                    prior_bp=prior_bp,
                    fitted_bp=projected_fitted_bp,
                    lambda_bp=lambda_bp,
                    blended_bp=blended_bp,
                )

                # blend_weights performs a linear interpolation only —
                # integer floor division and the canonical-order remainder
                # correction can each nudge an individual signal by a basis
                # point or two, which could in theory land just outside a
                # bound even though both inputs to the blend were themselves
                # within it. One more projection pass (same reused function,
                # zero adjustment) is the final safety net before this is
                # allowed into a WeightResolution.
                final_projection = project_adjustment(
                    blended_bp, bounds_bp, _ZERO_ADJUSTMENT, SIGNAL_ORDER,
                    max_iterations=config.max_projection_iterations,
                )
                if final_projection.success and final_projection.weights_bp is not None:
                    tracer.trace_stage(
                        "FINAL PROJECTION",
                        input_bp=blended_bp,
                        output_bp=final_projection.weights_bp,
                        sum=sum(final_projection.weights_bp.values()),
                    )
                    tracer.trace_stage(
                        "HISTORY",
                        route="fitted_blend",
                        fitted_weights_available=True,
                        history_admissibility=history_assessment.admissibility,
                    )
                    source = SOURCE_HISTORICALLY_FITTED if lambda_bp == TOTAL_BP else SOURCE_BLENDED
                    result = WeightResolution(
                        tenant_id=tenant_id,
                        source=source,
                        status=STATUS_RECOMMENDATION,
                        weights_bp=final_projection.weights_bp,
                        available_signals=availability_result.available_signals,
                        excluded_signals=availability_result.excluded_signals,
                        history_assessment=HistoryAssessment(
                            admissibility=history_assessment.admissibility,
                            reasons=history_assessment.reasons,
                            lambda_bp=lambda_bp,
                        ),
                        prior_bp=prior_bp,
                        adjustments_bp={signal: 0 for signal in SIGNAL_ORDER},
                        bounds_bp=bounds_bp,
                        confidence=lambda_bp / TOTAL_BP,  # derived from bp — trust, not a weight
                        evidence=(),
                        fallback_reasons=tuple(fallback_reasons),
                        requires_admin_approval=True,
                        prompt_version=COMBINED_PROMPT_VERSION,
                        transform_version=TRANSFORM_VERSION,
                        agent_version=AGENT_VERSION,
                        generated_at=generated_at,
                        fitted_projection_applied=fitted_projection_applied,
                        fitted_projection_changed_signals=fitted_projection_changed_signals,
                    )
                    self._audit_log(
                        result, fitted_available=True, llm_profile_used=False, llm_adjustment_used=False
                    )
                    tracer.trace_final_summary(result)
                    return result
                log.error(
                    "tenant %s post-blend projection could not satisfy current bounds: %s",
                    tenant_id, final_projection.reason,
                )
                tracer.trace_decision(
                    "FINAL PROJECTION", "reject_fitted", reason=FITTED_PROJECTION_INFEASIBLE
                )
                fallback_reasons.append(FITTED_PROJECTION_INFEASIBLE)
                fitted = None
            else:
                log.error(
                    "tenant %s fitted vector could not be projected into current bounds: %s",
                    tenant_id, fitted_projection.reason,
                )
                tracer.trace_decision(
                    "FITTED PROJECTION", "reject_fitted", reason=FITTED_PROJECTION_INFEASIBLE
                )
                fallback_reasons.append(FITTED_PROJECTION_INFEASIBLE)
                fitted = None

        if fitted is None:
            tracer.trace_stage(
                "HISTORY",
                route="cold_start",
                fitted_weights_available=fitted_provider_returned_data,
                history_admissibility=history_assessment.admissibility,
            )

        # --- 4.4 two-stage LLM-adjusted prior (cold start, Improvement 2) --
        llm_profile_used = False
        llm_adjustment_used = False
        if config.llm_enabled and tenant_description:
            try:
                extraction = TenantProfileExtractor(self._resolve_llm_provider()).extract(
                    tenant_description, config, tracer=tracer
                )
            except LLMAdjustmentError as exc:
                log.warning("tenant %s profile extraction failed: %s", tenant_id, exc)
                fallback_reasons.append(f"profile_extraction_failed: {exc}")
            else:
                llm_profile_used = True
                missing_critical = extraction.profile.missing_critical_fields()
                if missing_critical:
                    tracer.trace_decision(
                        "PROFILE GATE",
                        "fallback_to_prior",
                        reason="missing_critical_profile_fields",
                        profile_confidence=round(extraction.profile_confidence, 2),
                        min_required=config.min_profile_confidence,
                        critical_fields_complete=False,
                    )
                    fallback_reasons.append(
                        f"critical_profile_fields_missing: {sorted(missing_critical)}"
                    )
                elif extraction.profile_confidence < config.min_profile_confidence:
                    tracer.trace_decision(
                        "PROFILE GATE",
                        "fallback_to_prior",
                        reason="low_profile_confidence",
                        profile_confidence=round(extraction.profile_confidence, 2),
                        min_required=config.min_profile_confidence,
                        critical_fields_complete=True,
                    )
                    fallback_reasons.append(
                        f"low_profile_confidence: {extraction.profile_confidence:.2f} < "
                        f"{config.min_profile_confidence:.2f}"
                    )
                else:
                    tracer.trace_decision(
                        "PROFILE GATE",
                        "proceed_to_adjustment",
                        profile_confidence=round(extraction.profile_confidence, 2),
                        min_required=config.min_profile_confidence,
                        critical_fields_complete=True,
                    )
                    try:
                        proposal = WeightAdjustmentGenerator(self._resolve_llm_provider()).propose(
                            extraction.profile, prior_bp, bounds_bp, config, tracer=tracer
                        )
                    except LLMAdjustmentError as exc:
                        log.warning("tenant %s weight adjustment failed: %s", tenant_id, exc)
                        fallback_reasons.append(f"weight_adjustment_failed: {exc}")
                    else:
                        llm_adjustment_used = True
                        outcome = project_adjustment(
                            prior_bp,
                            bounds_bp,
                            proposal.adjustment_bp,
                            SIGNAL_ORDER,
                            max_iterations=config.max_projection_iterations,
                        )
                        if outcome.success and outcome.weights_bp is not None:
                            naive_bp = {
                                signal: prior_bp[signal] + proposal.adjustment_bp[signal]
                                for signal in SIGNAL_ORDER
                            }
                            tracer.trace_stage(
                                "PROJECTION",
                                input_prior_bp=prior_bp,
                                requested_adjustment_bp=proposal.adjustment_bp,
                                projected_weights_bp=outcome.weights_bp,
                                changed_signals=[
                                    s for s in SIGNAL_ORDER if outcome.weights_bp[s] != naive_bp[s]
                                ],
                                sum=sum(outcome.weights_bp.values()),
                            )
                            if extraction.retried or proposal.retried:
                                fallback_reasons.append("llm_retry_used")
                            result = WeightResolution(
                                tenant_id=tenant_id,
                                source=SOURCE_LLM_ADJUSTED_PRIOR,
                                status=STATUS_RECOMMENDATION,
                                weights_bp=outcome.weights_bp,
                                available_signals=availability_result.available_signals,
                                excluded_signals=availability_result.excluded_signals,
                                history_assessment=history_assessment,
                                prior_bp=prior_bp,
                                adjustments_bp=dict(proposal.adjustment_bp),
                                bounds_bp=bounds_bp,
                                confidence=extraction.profile_confidence,
                                evidence=proposal.evidence,
                                fallback_reasons=tuple(fallback_reasons),
                                requires_admin_approval=True,
                                prompt_version=COMBINED_PROMPT_VERSION,
                                transform_version=TRANSFORM_VERSION,
                                agent_version=AGENT_VERSION,
                                generated_at=generated_at,
                            )
                            self._audit_log(
                                result,
                                fitted_available=fitted_provider_returned_data,
                                llm_profile_used=llm_profile_used,
                                llm_adjustment_used=llm_adjustment_used,
                            )
                            tracer.trace_final_summary(result)
                            return result
                        tracer.trace_decision(
                            "PROJECTION", "projection_non_convergence", reason=outcome.reason
                        )
                        fallback_reasons.append(f"projection_non_convergence: {outcome.reason}")
        elif not config.llm_enabled:
            fallback_reasons.append("llm_disabled")
        else:
            fallback_reasons.append("no_tenant_description_supplied")

        # --- 4.5 fallback -----------------------------------------------------
        result = WeightResolution(
            tenant_id=tenant_id,
            source=SOURCE_PRIOR,
            status=STATUS_RECOMMENDATION,
            weights_bp=dict(prior_bp),
            available_signals=availability_result.available_signals,
            excluded_signals=availability_result.excluded_signals,
            history_assessment=history_assessment,
            prior_bp=prior_bp,
            adjustments_bp={signal: 0 for signal in SIGNAL_ORDER},
            bounds_bp=bounds_bp,
            confidence=0.0,
            evidence=(),
            fallback_reasons=tuple(fallback_reasons),
            requires_admin_approval=True,
            prompt_version=COMBINED_PROMPT_VERSION,
            transform_version=TRANSFORM_VERSION,
            agent_version=AGENT_VERSION,
            generated_at=generated_at,
        )
        self._audit_log(
            result,
            fitted_available=fitted_provider_returned_data,
            llm_profile_used=llm_profile_used,
            llm_adjustment_used=llm_adjustment_used,
        )
        tracer.trace_final_summary(result)
        return result

    @staticmethod
    def _audit_log(
        result: WeightResolution,
        *,
        fitted_available: bool,
        llm_profile_used: bool,
        llm_adjustment_used: bool,
    ) -> None:
        """One structured line per resolution (Improvement 9). Never logs raw
        tenant description, raw prompt text, or any secret — only enum-like
        codes and counts. Reaching this call at all means
        ``WeightResolution.__post_init__`` already accepted ``result``, so
        ``validation_result`` is always "passed" here — a resolver bug that
        produced an invalid result would have raised before this line runs.
        """
        log.info(
            "weight_agent.resolve tenant_id=%s source=%s history_admissibility=%s "
            "fitted_weights_available=%s fitted_projection_applied=%s "
            "llm_profile_used=%s llm_adjustment_used=%s "
            "excluded_signal_count=%d fallback_reasons=%s validation_result=passed "
            "prompt_version=%s agent_version=%s",
            result.tenant_id,
            result.source,
            result.history_assessment.admissibility,
            fitted_available,
            result.fitted_projection_applied,
            llm_profile_used,
            llm_adjustment_used,
            len(result.excluded_signals),
            list(result.fallback_reasons),
            result.prompt_version,
            result.agent_version,
        )
