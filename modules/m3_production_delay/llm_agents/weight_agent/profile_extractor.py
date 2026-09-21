"""§7 tenant profile extraction — stage 1 of the two-stage cold-start path
(Improvement 2). This is the ONLY place raw tenant prose is read. Its output,
:class:`TenantProfile`, is a closed-vocabulary, self-validating value object
(see models.py) — the only thing stage 2 (:mod:`weight_adjustment`) is ever
given. A prompt injection in the raw description has no path past this stage:
it can only ever change which of a fixed set of enum values gets extracted,
never anything a downstream numeric stage reads as free text.

``profile_confidence`` is computed HERE, deterministically, from how many
fields the validated profile actually carries — never self-reported by the
model that also proposes adjustments (that would let it score its own
eligibility). See models.TenantProfile.missing_critical_fields for the
harder gate the resolver applies on top of this.
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
from m3_production_delay.llm_agents.weight_agent.models import PROFILE_FIELDS, TenantProfile
from m3_production_delay.llm_agents.weight_agent.tracing import NOOP_TRACER, WeightAgentTracer
from m3_production_delay.prompt import load_prompt_template

log = get_logger("m3_production_delay.weight_agent.profile_extractor")

PROMPT_VERSION = "profile-v1"

# Loaded once at import time from modules/m3_production_delay/prompt/ — the
# prompt's wording lives there, not here, so it can be reviewed/edited
# without touching this module.
_PROMPT_TEMPLATE = Template(load_prompt_template("profile_extraction.txt"))

# Fixed so a retry reuses the identical request (spec: "at most one retry,
# same seed and temperature"). See ports.GenerationConfig's docstring for why
# this narrows variance rather than promising byte-identical provider output.
_GENERATION_CONFIG = GenerationConfig(temperature=0.0, seed=0)


@dataclass(frozen=True)
class ProfileExtractionResult:
    profile: TenantProfile
    profile_confidence: float
    warnings: tuple[str, ...]  # e.g. "field X: out-of-enum value ignored" — no raw text
    retried: bool


def sanitize_description(description: str, max_chars: int) -> tuple[str, bool]:
    """Truncate to the configured limit. Returns (text, was_truncated)."""
    stripped = description.strip()
    if len(stripped) <= max_chars:
        return stripped, False
    return stripped[:max_chars], True


def _build_prompt(description: str) -> str:
    fields_spec = "\n".join(
        f"  - {name}: one of {sorted(values)} or null" for name, values in PROFILE_FIELDS.items()
    )
    return _PROMPT_TEMPLATE.substitute(fields_spec=fields_spec, description=description)


def _parse_response(raw_text: str) -> tuple[TenantProfile, float, tuple[str, ...]]:
    try:
        payload = json.loads(unwrap_json_code_fence(raw_text))
    except json.JSONDecodeError as exc:
        raise LLMAdjustmentError(f"invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMAdjustmentError("schema_mismatch: response is not a JSON object")

    raw_profile = payload.get("profile")
    fields: dict[str, str | None] = dict.fromkeys(PROFILE_FIELDS, None)
    warnings: list[str] = []
    if isinstance(raw_profile, dict):
        for name, allowed in PROFILE_FIELDS.items():
            value = raw_profile.get(name)
            if value is None:
                continue
            if isinstance(value, str) and value in allowed:
                fields[name] = value
            else:
                # Out-of-enum values must not silently disappear (Improvement
                # 9) — recorded as a short, non-sensitive diagnostic (field
                # name only; the LLM's own short classification token, not
                # tenant free text) rather than dropped with no trace.
                warnings.append(f"{name}: out-of-enum value ignored")
    elif raw_profile is not None:
        warnings.append("profile: not a JSON object, ignored entirely")

    profile = TenantProfile(**fields)
    extracted = sum(1 for value in fields.values() if value is not None)
    confidence = extracted / len(PROFILE_FIELDS)
    return profile, confidence, tuple(warnings)


class TenantProfileExtractor:
    """Wraps one :class:`LLMProvider` call. At most one retry, identical
    request both times."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def extract(
        self,
        description: str,
        config: WeightAgentConfig,
        *,
        tracer: WeightAgentTracer = NOOP_TRACER,
    ) -> ProfileExtractionResult:
        clean_description, truncated = sanitize_description(
            description, config.max_tenant_description_chars
        )
        if truncated:
            log.info(
                "tenant description truncated to %d chars before profile extraction",
                config.max_tenant_description_chars,
            )
        prompt = _build_prompt(clean_description)

        tracer.trace_stage(
            "LLM CALL #1",
            stage="tenant_profile_extraction",
            provider=type(self._provider).__name__,
            prompt_version=PROMPT_VERSION,
            generation_temperature=_GENERATION_CONFIG.temperature,
            generation_seed=_GENERATION_CONFIG.seed,
        )
        # Content-gated inside trace_prompt itself — this is a no-op unless
        # trace-content is explicitly on. What it shows (or doesn't) is the
        # ACTUAL prompt sent, description already embedded by _build_prompt.
        tracer.trace_prompt("PROFILE EXTRACTION", prompt=prompt)

        last_error = LLMAdjustmentError("profile_extraction_failed: unreachable default")
        for attempt in range(2):
            if attempt == 1:
                tracer.trace_stage("LLM CALL #1 RETRY", attempt=2)
            try:
                raw_text = self._provider.generate(
                    prompt, max_tokens=600, generation_config=_GENERATION_CONFIG
                )
            except Exception as exc:  # provider timeout / transport failure
                last_error = LLMAdjustmentError(f"provider_error: {exc}")
                log.warning("profile extraction call failed (attempt %d): %s", attempt + 1, exc)
                tracer.trace_decision(
                    "PROFILE EXTRACTION RESPONSE", "provider_error", retry_attempt=attempt + 1
                )
                continue
            tracer.trace_stage(
                "PROFILE EXTRACTION RESPONSE", response_length=len(raw_text), retry_attempt=attempt + 1
            )
            tracer.trace_response("PROFILE EXTRACTION", raw_text)
            try:
                profile, confidence, warnings = _parse_response(raw_text)
            except LLMAdjustmentError as exc:
                last_error = exc
                log.warning("profile extraction response rejected (attempt %d): %s", attempt + 1, exc)
                tracer.trace_decision(
                    "PROFILE EXTRACTION RESPONSE",
                    "parse_failure",
                    parse_success=False,
                    retrying=attempt == 0,
                )
                continue
            if attempt == 1:
                log.info("profile extraction succeeded on retry")
            for warning in warnings:
                log.debug("profile extraction warning: %s", warning)
            tracer.trace_decision("PROFILE EXTRACTION RESPONSE", "parse_success", parse_success=True)
            tracer.trace_stage(
                "PROFILE PARSED",
                industry=profile.industry,
                production_type=profile.production_type,
                material_dependency=profile.material_dependency,
                supplier_dependency=profile.supplier_dependency,
                workforce_dependency=profile.workforce_dependency,
                seasonality_level=profile.seasonality_level,
                profile_confidence=round(confidence, 2),
                critical_fields_complete=not profile.missing_critical_fields(),
                warnings=list(warnings),
            )
            return ProfileExtractionResult(
                profile=profile,
                profile_confidence=confidence,
                warnings=warnings,
                retried=attempt == 1,
            )

        tracer.trace_decision("PROFILE EXTRACTION RESPONSE", "parse_failure", parse_success=False, retrying=False)
        raise last_error
