"""M3 Production Delay orchestrator.

Thin composition boundary between a caller (a future Configurator endpoint,
this module's own CLI runner below, or eventually the rule/risk engines) and
this module's LLM agents (``llm_agents/``). Owns no statistical logic and no
IO of its own — it builds a request, calls an agent, and returns the result
unchanged.

Two agents are composed here:

* **Weight Agent** (``llm_agents/weight_agent/``) — resolves the tenant's
  risk-signal weight vector. Weight validation, history-floor computation,
  shrinkage, projection, LLM profile extraction/adjustment and fitted-weight
  compatibility all remain inside that package.
* **Review Agent** (``llm_agents/review_agent/``) — judges whether each
  composed delay-insight line is supported by the Risk Engine's evidence.
  Evidence building, line composition, the deterministic validators and the
  writeback remain inside ``review/``; this class only hands the agent to the
  review pipeline.

Both take the same optional ``llm_provider``, so one injected provider covers
every LLM call M3 makes — which is what lets a test drive the whole module
with a scripted provider and no network.

Naming note: this file was named ``occustrator.py`` in early scaffolding — a
one-line placeholder comment, never committed to git history and never
imported by anything. It was already replaced by this correctly-spelled
``orchestrator.py`` (still just a placeholder) before this integration work
started, so nothing here is a rename of shipped, tracked, or referenced code.

Deferred boundaries — explicitly NOT implemented here, and not faked:
rule engine, risk engine, evaluation agent, historical fitting pipeline
(the ``FittedWeightsProvider`` below is a fixed local fixture for manual
testing only), dashboard, new authentication, new tenant resolution. Tenant
identity must arrive from an already-trusted upstream context; it is never
derived from ``tenant_description`` or any LLM output.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from maxxflow_core.errors import get_logger
from maxxflow_core.ports import GenerationConfig, LLMProvider
from maxxflow_core.settings import get_settings

from m3_production_delay.llm_agents.review_agent.config import ReviewAgentConfig
from m3_production_delay.llm_agents.review_agent.resolver import ReviewAgent
from m3_production_delay.llm_agents.weight_agent.config import WeightAgentConfig
from m3_production_delay.llm_agents.weight_agent.exceptions import AllSignalsUnavailableError
from m3_production_delay.llm_agents.weight_agent.models import (
    SIGNAL_ORDER,
    FittedWeights,
    HistoryInputs,
    SignalName,
    WeightResolution,
)
from m3_production_delay.llm_agents.weight_agent.providers import FittedWeightsProvider
from m3_production_delay.llm_agents.weight_agent.resolver import WeightAgent
from m3_production_delay.llm_agents.weight_agent.tracing import WeightAgentTracer
from m3_production_delay.rule_engine.elements import weights_bp_to_risk_weights
from m3_production_delay.tenant_context_reader import TenantContextReader

if TYPE_CHECKING:  # review/ imports this module inside a function, never at
    # import time — keeping the annotation import-only preserves that.
    from m3_production_delay.review.schemas import ValidatedInsight

log = get_logger("m3_production_delay.orchestrator")


@dataclass(frozen=True)
class WeightAgentRequest:
    """Bundles exactly the inputs ``WeightAgent.resolve()`` accepts (spec
    §4/§9) — a call-boundary convenience, not a second validation schema.
    ``WeightAgent`` and its domain models already own every invariant these
    fields are subject to; this class re-checks nothing.

    ``tenant_id`` must already be a trusted identifier resolved by the
    caller's own security/session context — never derived here from
    ``tenant_description``, ``request_headers``, or any LLM output.

    ``tenant_description`` is an explicit override: when supplied, it is used
    as-is (unchanged behaviour for any existing caller). When left ``None``,
    the orchestrator resolves one via :class:`TenantContextReader` instead —
    see :meth:`ProductionDelayOrchestrator.resolve_weights`. ``request_headers``
    only ever feeds that resolution's request-header source; it can never
    change ``tenant_id`` or anything else on this request.
    """

    tenant_id: str
    availability: dict[str, bool]
    configured_bp: dict[str, int] | None = None
    exclusion_reasons: dict[str, str] | None = None
    tenant_description: str | None = None
    history_inputs: HistoryInputs | None = None
    request_headers: Mapping[str, str] | None = None
    # Reuse an existing upstream request/correlation id if the caller already
    # has one; otherwise resolve_weights() generates a fresh uuid4. Never the
    # tenant id — the trace id exists to disambiguate concurrent requests for
    # the SAME tenant, which a tenant id cannot do.
    trace_id: str | None = None


class ProductionDelayOrchestrator:
    """Composes the M3 agents.

    Configured-weight priority, availability handling, historical blending,
    fitted-weight bounds projection, and the two-stage LLM cold start are all
    the Weight Agent's own responsibility (spec §4) — this class never
    reimplements or shortcuts any of it, and never persists a recommendation
    as approved tenant configuration. The same restraint applies to the
    Review Agent: this class hands it to the review pipeline and returns what
    comes back, and never writes, reorders or vets an explanation line itself.
    """

    def __init__(
        self,
        *,
        fitted_provider: FittedWeightsProvider | None = None,
        llm_provider: LLMProvider | None = None,
        config: WeightAgentConfig | None = None,
        tenant_context_reader: TenantContextReader | None = None,
        review_agent: ReviewAgent | None = None,
        review_config: ReviewAgentConfig | None = None,
    ) -> None:
        self._weight_agent = WeightAgent(
            fitted_provider=fitted_provider, llm_provider=llm_provider, config=config
        )
        # Same llm_provider as the Weight Agent by default: one injected
        # provider then covers every LLM call M3 makes. A fully-built
        # `review_agent` still wins, for a caller that needs the two agents on
        # different providers or configs.
        self._review_agent = review_agent or ReviewAgent(
            llm_provider=llm_provider, config=review_config
        )
        self._tenant_context_reader = tenant_context_reader or TenantContextReader()

    @property
    def review_agent(self) -> ReviewAgent:
        return self._review_agent

    def resolve_weights(self, request: WeightAgentRequest) -> WeightResolution:
        """Returns the Weight Agent's ``WeightResolution`` unchanged.

        ``AllSignalsUnavailableError`` propagates unchanged — there is no
        valid weight vector to substitute for it, so the caller must decide
        what "cannot score this tenant" means for it. Any other exception is
        equally left to propagate: only that one expected domain error is
        ever meaningful to translate, and this class does not even do that
        much, to guarantee an unexpected failure can never be mistaken for a
        successful resolution.
        """
        log.info(
            "m3_orchestrator operation=resolve_weights tenant_id=%s stage=start",
            request.tenant_id,
        )
        settings = get_settings()
        tracer = WeightAgentTracer(
            trace_id=request.trace_id or str(uuid.uuid4()),
            enabled=settings.m3_weight_trace_enabled,
            include_content=settings.m3_weight_trace_include_content,
        )
        tracer.trace_stage(
            "START",
            tenant_id=request.tenant_id,
            configured_weights_present=request.configured_bp is not None,
            history_inputs_present=request.history_inputs is not None,
            request_headers_present=request.request_headers is not None,
        )

        tenant_description = request.tenant_description
        if tenant_description is None:
            # No caller-supplied description — resolve one ourselves. The
            # Weight Agent never sees which of the three sources (or none)
            # produced it; it only ever receives the resulting string or None.
            resolved = self._tenant_context_reader.resolve_description(
                request.tenant_id, headers=request.request_headers, tracer=tracer
            )
            tenant_description = resolved.description
        result = self._weight_agent.resolve(
            request.tenant_id,
            availability=request.availability,
            configured_bp=request.configured_bp,
            exclusion_reasons=request.exclusion_reasons,
            tenant_description=tenant_description,
            history_inputs=request.history_inputs,
            tracer=tracer,
        )
        log.info(
            "m3_orchestrator operation=resolve_weights tenant_id=%s stage=done "
            "source=%s status=%s",
            request.tenant_id,
            result.source,
            result.status,
        )
        return result

    def resolve_risk_weights(self, request: WeightAgentRequest) -> dict[str, float]:
        """Resolves this tenant's weights via the Weight Agent, then adapts
        them into the rule engine's `risk_weights` shape (see
        `rule_engine.elements.weights_bp_to_risk_weights`), ready to pass
        straight into `rule_engine.elements.calculate_delay_elements_for_jobs`.

        `resolve_weights()` above stays an unchanged pass-through to
        `WeightAgent.resolve()` (see its own docstring); this method is the
        one boundary where that result's basis-point signal vocabulary gets
        translated into the rule engine's own key names and float scale -
        the rule engine itself never depends on the Weight Agent's types.
        """
        result = self.resolve_weights(request)
        risk_weights = weights_bp_to_risk_weights(result.weights_bp)
        log.info(
            "m3_weight_resolution tenant_id=%s source=%s status=%s weights_bp=%s "
            "risk_weights=%s excluded_signals=%s fallback_reasons=%s",
            request.tenant_id,
            result.source,
            result.status,
            result.weights_bp,
            risk_weights,
            [excluded.signal for excluded in result.excluded_signals],
            result.fallback_reasons,
        )
        return risk_weights

    def review_jobs(
        self,
        scored_jobs: list[dict],
        *,
        weights: dict[str, float],
        threshold: float,
    ) -> "list[ValidatedInsight]":
        """Runs the Review Agent over already-scored jobs and returns one
        :class:`ValidatedInsight` per job.

        ``scored_jobs`` is ``calculate_delay_elements_for_jobs()``'s output,
        ``weights`` the vector that produced those scores (so an explanation
        can never be attributed against a different weighting than the one
        scored), and ``threshold`` the tenant's delay cutoff — required, since
        the same composite score means different things per tenant and there
        is no calibrated default anywhere in M3.

        No IO happens here: reading the tenant, rolling up and writing the
        advisory back all belong to ``review/pipeline.py``, the same way this
        class never reads a tenant to resolve weights.
        """
        from m3_production_delay.review.pipeline import review_jobs as _review_jobs

        log.info(
            "m3_orchestrator operation=review_jobs jobs=%d threshold=%s stage=start",
            len(scored_jobs),
            threshold,
        )
        insights = _review_jobs(scored_jobs, weights, threshold, self._review_agent)
        log.info(
            "m3_orchestrator operation=review_jobs jobs=%d stage=done", len(insights)
        )
        return insights


# ---------------------------------------------------------------------------
# Manual local runner (spec §9-§15): `uv run python -m m3_production_delay
# .orchestrator --scenario <name>`. Everything below this line exists only to
# exercise the orchestrator above from the command line; it is not part of
# the orchestrator's own public surface and services/configurator (or any
# other caller) should import the class above directly instead.
# ---------------------------------------------------------------------------


class _DemoFittedWeightsProvider:
    """Fixed in-memory ``FittedWeights`` for local manual runs only
    (``--scenario fitted``). NOT a production historical fitting pipeline —
    none exists yet (see module docstring) — and never registered as a
    default anywhere outside this CLI runner.
    """

    def __init__(self, fitted: FittedWeights) -> None:
        self._fitted = fitted

    def get(self, tenant_id: str) -> FittedWeights | None:
        return self._fitted


class _DemoScriptedLLMProvider:
    """DEMO ONLY — NOT PRODUCTION. Selected only via ``--demo-llm``, never by
    ``maxxflow_providers``' production factory. The real ``StubLLMProvider``
    is deterministic but returns text, not JSON, so the Weight Agent's
    two-stage cold start always falls back to the prior against it; this
    scripted provider returns fixed, schema-valid JSON for exactly the two
    calls the Weight Agent ever makes (profile extraction, then weight
    adjustment), so the full successful flow can be watched end to end
    locally without a paid external LLM.
    """

    _PROFILE_RESPONSE = json.dumps(
        {
            "profile": {
                "production_type": "make_to_order",
                "material_dependency": "high",
                "supplier_dependency": "high",
                "workforce_dependency": "medium",
                "seasonality_level": "low",
            }
        }
    )
    _ADJUSTMENT_RESPONSE = json.dumps(
        {
            "adjustments_bp": {
                "time_overrun": 200,
                "operator_skill": -200,
                "seasonality": 0,
                "material_availability": 0,
                "supplier_reliability": 0,
            },
            "evidence": ["DEMO ONLY: high material/supplier dependency raises time_overrun weight"],
        }
    )

    name = "demo-scripted (NOT PRODUCTION)"

    def __init__(self) -> None:
        self._call_count = 0

    def generate(
        self, prompt: str, *, max_tokens: int = 256, generation_config: GenerationConfig | None = None
    ) -> str:
        self._call_count += 1
        return self._PROFILE_RESPONSE if self._call_count == 1 else self._ADJUSTMENT_RESPONSE


_SCENARIOS = ("configured", "cold-start", "missing-signal", "no-signals", "fitted")


def _build_request(
    scenario: str, tenant_id: str, *, tenant_description: str | None = None
) -> WeightAgentRequest:
    all_available = {signal: True for signal in SIGNAL_ORDER}

    if scenario == "configured":
        return WeightAgentRequest(
            tenant_id=tenant_id,
            availability=all_available,
            configured_bp={
                "time_overrun": 4000,
                "operator_skill": 3000,
                "seasonality": 1000,
                "material_availability": 1500,
                "supplier_reliability": 500,
            },
        )
    if scenario == "cold-start":
        # No --tenant-description given: tenant_description stays None, so
        # ProductionDelayOrchestrator resolves one via TenantContextReader
        # (request header -> database -> local YAML). Pass --tenant-description
        # to bypass that chain exactly like an existing trusted caller would.
        return WeightAgentRequest(
            tenant_id=tenant_id,
            availability=all_available,
            tenant_description=tenant_description,
        )
    if scenario == "missing-signal":
        availability = dict(all_available)
        availability["seasonality"] = False
        return WeightAgentRequest(
            tenant_id=tenant_id,
            availability=availability,
            exclusion_reasons={"seasonality": "no_seasonal_data_collected"},
        )
    if scenario == "no-signals":
        return WeightAgentRequest(
            tenant_id=tenant_id,
            availability={signal: False for signal in SIGNAL_ORDER},
            exclusion_reasons={signal: "no_computable_signals" for signal in SIGNAL_ORDER},
        )
    if scenario == "fitted":
        return WeightAgentRequest(
            tenant_id=tenant_id,
            availability=all_available,
            history_inputs=HistoryInputs(
                history_span_days=400, completed_work_orders=800, delayed_work_orders=120
            ),
        )
    raise SystemExit(f"unknown scenario {scenario!r}; choose from {_SCENARIOS}")


def _build_orchestrator(scenario: str, *, demo_llm: bool) -> ProductionDelayOrchestrator:
    llm_provider = _DemoScriptedLLMProvider() if demo_llm else None
    if scenario != "fitted":
        return ProductionDelayOrchestrator(llm_provider=llm_provider)
    # Deliberately the exact even-split example from the bounds-projection
    # fix: valid FittedWeights, but violates this tenant's configured Bounds
    # (time_overrun min=3000, supplier_reliability max=1500), so this
    # scenario also demonstrates fitted_projection_applied end to end.
    fitted = FittedWeights(
        weights_bp={signal: 2000 for signal in SIGNAL_ORDER},
        # SIGNAL_ORDER is `tuple[str, ...]` (its values happen to be exactly
        # the SignalName literals, but the constant itself isn't narrowed) —
        # cast documents that fact instead of widening FittedWeights' own
        # field type to accept a plain str frozenset.
        signal_set=cast("frozenset[SignalName]", frozenset(SIGNAL_ORDER)),
        delayed_event_count=500,
    )
    return ProductionDelayOrchestrator(
        fitted_provider=_DemoFittedWeightsProvider(fitted), llm_provider=llm_provider
    )


def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="m3_production_delay.orchestrator",
        description="Manually resolve M3 production-delay weights for one local scenario.",
    )
    parser.add_argument("--scenario", required=True, choices=_SCENARIOS)
    parser.add_argument("--tenant-id", default="demo-tenant")
    parser.add_argument(
        "--tenant-description",
        default=None,
        help="cold-start only: override the resolved description directly, "
        "bypassing TenantContextReader (request header / database / local YAML)",
    )
    parser.add_argument(
        "--trace", action="store_true", help="emit the local debug execution trace (off by default)"
    )
    parser.add_argument(
        "--trace-content",
        action="store_true",
        help="also emit raw tenant description / LLM prompt / LLM response text (off by "
        "default); implies --trace",
    )
    parser.add_argument(
        "--demo-llm",
        action="store_true",
        help="DEMO ONLY: use a scripted provider returning valid profile/adjustment JSON "
        "instead of the default StubLLMProvider, to exercise the full two-LLM-call flow "
        "without a paid external LLM",
    )
    args = parser.parse_args(argv)

    # CLI flags map to the same Settings fields a real deployment would set
    # via environment — set before the first get_settings() call (Settings is
    # lru_cached), exactly like the existing MLflow-URI CLI override in
    # maxxflow_cli.__main__._redirect_registry. The Weight Agent and
    # orchestrator never read os.environ directly; only this CLI script does.
    if args.trace or args.trace_content:
        os.environ["M3_WEIGHT_AGENT_TRACE_ENABLED"] = "true"
    if args.trace_content:
        os.environ["M3_WEIGHT_AGENT_TRACE_INCLUDE_CONTENT"] = "true"
    if args.trace or args.trace_content:
        get_settings.cache_clear()

    orchestrator = _build_orchestrator(args.scenario, demo_llm=args.demo_llm)
    request = _build_request(args.scenario, args.tenant_id, tenant_description=args.tenant_description)
    try:
        result = orchestrator.resolve_weights(request)
    except AllSignalsUnavailableError as exc:
        print(
            json.dumps(
                {"status": "unavailable", "reason": "no_computable_signals", "detail": str(exc)},
                indent=2,
            )
        )
        return 1
    # Trace output (if any) already went to the trace logger above this line;
    # this JSON print is the same, unchanged normal CLI output as before —
    # kept visually separate from the trace banner rather than merged into it.
    print(json.dumps(result.to_json_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
