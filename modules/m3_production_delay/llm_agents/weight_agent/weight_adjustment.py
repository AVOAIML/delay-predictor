"""§4.4/§8 bounded weight adjustment — stage 2 of the two-stage cold-start
path (Improvement 2). This is the trust boundary: :func:`propose_adjustment`
takes a validated :class:`TenantProfile`, never a raw string. There is no
parameter here a prompt injection in the original tenant description could
reach — the type signature itself is the enforcement, not just careful
prompt phrasing.

The LLM may: propose a bounded, sum-to-zero adjustment and write
human-readable evidence strings. It may not: return final weights, name a
tenant, or touch bounds/prior/normalisation — those are computed by
resolver.py and projection.py, never read from this stage's response beyond
``adjustments_bp`` and ``evidence``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from string import Template

from maxxflow_core.errors import get_logger
from maxxflow_core.ports import GenerationConfig, LLMProvider

from m3_production_delay.llm_agents.weight_agent.config import WeightAgentConfig
from m3_production_delay.llm_agents.weight_agent.exceptions import LLMAdjustmentError
from m3_production_delay.llm_agents.weight_agent.json_response import unwrap_json_code_fence
from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER, SIGNAL_SET, Bounds, TenantProfile
from m3_production_delay.llm_agents.weight_agent.tracing import NOOP_TRACER, WeightAgentTracer
from m3_production_delay.prompt import load_prompt_template

log = get_logger("m3_production_delay.weight_agent.weight_adjustment")

PROMPT_VERSION = "adjustment-v2"

# Loaded once at import time from modules/m3_production_delay/prompt/ — the
# prompt's wording lives there, not here, so it can be reviewed/edited
# without touching this module.
_PROMPT_TEMPLATE = Template(load_prompt_template("weight_adjustment.txt"))

# Fixed so a retry reuses the identical request (spec: "at most one retry,
# same seed and temperature").
_GENERATION_CONFIG = GenerationConfig(temperature=0.0, seed=0)


@dataclass(frozen=True)
class AdjustmentProposal:
    adjustment_bp: dict[str, int]
    evidence: tuple[str, ...]
    retried: bool


def _build_prompt(profile: TenantProfile, prior_bp: dict[str, int], bounds_bp: dict[str, Bounds]) -> str:
    profile_spec = "\n".join(
        f"  - {name}: {value if value is not None else 'unknown'}"
        for name, value in profile.fields().items()
    )
    bounds_spec = "\n".join(
        f"  - {signal}: prior={prior_bp[signal]}, min={bounds_bp[signal].min}, "
        f"max={bounds_bp[signal].max}"
        for signal in SIGNAL_ORDER
    )
    return _PROMPT_TEMPLATE.substitute(profile_spec=profile_spec, bounds_spec=bounds_spec)


def _validate_adjustment_bp(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise LLMAdjustmentError("schema_mismatch: adjustments_bp is not an object")
    keys = set(raw)
    if keys != SIGNAL_SET:
        raise LLMAdjustmentError(
            f"out_of_enum_signal: adjustments_bp keys {sorted(keys)} do not match "
            f"{sorted(SIGNAL_ORDER)}"
        )
    adjustment_bp: dict[str, int] = {}
    for signal, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise LLMAdjustmentError(
                f"schema_mismatch: adjustments_bp[{signal!r}]={value!r} is not an integer"
            )
        adjustment_bp[signal] = value
    if sum(adjustment_bp.values()) != 0:
        raise LLMAdjustmentError(
            f"non_zero_sum_adjustment: adjustments_bp sums to "
            f"{sum(adjustment_bp.values())}, not 0"
        )
    return adjustment_bp


def _parse_evidence(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw if isinstance(item, str))[:20]


def _parse_response(raw_text: str) -> tuple[dict[str, int], tuple[str, ...]]:
    try:
        payload = json.loads(unwrap_json_code_fence(raw_text))
    except json.JSONDecodeError as exc:
        raise LLMAdjustmentError(f"invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMAdjustmentError("schema_mismatch: response is not a JSON object")
    adjustment_bp = _validate_adjustment_bp(payload.get("adjustments_bp"))
    evidence = _parse_evidence(payload.get("evidence"))
    return adjustment_bp, evidence


class WeightAdjustmentGenerator:
    """Wraps one :class:`LLMProvider` call. At most one retry, identical
    request both times. Never accepts a raw tenant description — only a
    validated :class:`TenantProfile`."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def propose(
        self,
        profile: TenantProfile,
        prior_bp: dict[str, int],
        bounds_bp: dict[str, Bounds],
        config: WeightAgentConfig,
        *,
        tracer: WeightAgentTracer = NOOP_TRACER,
    ) -> AdjustmentProposal:
        prompt = _build_prompt(profile, prior_bp, bounds_bp)

        tracer.trace_stage(
            "LLM CALL #2",
            stage="weight_adjustment",
            provider=type(self._provider).__name__,
            prompt_version=PROMPT_VERSION,
            generation_temperature=_GENERATION_CONFIG.temperature,
            generation_seed=_GENERATION_CONFIG.seed,
        )
        # _build_prompt only ever receives `profile` (validated, closed-
        # vocabulary), prior_bp and bounds_bp — never the raw tenant
        # description. Tracing the actual prompt here (content-gated) is
        # what makes that trust boundary visually verifiable, not something
        # this trace call enforces itself.
        tracer.trace_prompt("WEIGHT ADJUSTMENT", prompt=prompt)

        last_error = LLMAdjustmentError("weight_adjustment_failed: unreachable default")
        for attempt in range(2):
            if attempt == 1:
                tracer.trace_stage("LLM CALL #2 RETRY", attempt=2)
            try:
                raw_text = self._provider.generate(
                    prompt, max_tokens=400, generation_config=_GENERATION_CONFIG
                )
            except Exception as exc:  # provider timeout / transport failure
                last_error = LLMAdjustmentError(f"provider_error: {exc}")
                log.warning("weight adjustment call failed (attempt %d): %s", attempt + 1, exc)
                tracer.trace_decision(
                    "ADJUSTMENT VALIDATION", "rejected", fallback_reason=f"provider_error: {exc}"
                )
                continue
            tracer.trace_stage(
                "WEIGHT ADJUSTMENT RESPONSE", response_length=len(raw_text), retry_attempt=attempt + 1
            )
            tracer.trace_response("WEIGHT ADJUSTMENT", raw_text)
            try:
                adjustment_bp, evidence = _parse_response(raw_text)
            except LLMAdjustmentError as exc:
                last_error = exc
                log.warning("weight adjustment response rejected (attempt %d): %s", attempt + 1, exc)
                tracer.trace_decision("ADJUSTMENT VALIDATION", "rejected", fallback_reason=str(exc))
                continue
            if attempt == 1:
                log.info("weight adjustment succeeded on retry")
            tracer.trace_stage("ADJUSTMENT PARSED", adjustments_bp=adjustment_bp, evidence=list(evidence))
            tracer.trace_decision(
                "ADJUSTMENT VALIDATION", "accepted", known_signals=True, sum_rule=True, finite_values=True
            )
            return AdjustmentProposal(adjustment_bp=adjustment_bp, evidence=evidence, retried=attempt == 1)

        raise last_error
