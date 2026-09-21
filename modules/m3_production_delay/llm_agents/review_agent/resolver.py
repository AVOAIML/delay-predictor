"""The Review Agent — one LLM call, asked only whether each already-composed
explanation line is supported by the evidence.

Public entry point: :class:`ReviewAgent`. Same shape as
:class:`~m3_production_delay.llm_agents.weight_agent.resolver.WeightAgent` —
an optional ``llm_provider`` injected for tests, otherwise resolved from the
platform factory; an optional ``config``, otherwise the cached singleton; and
a tracer passed explicitly rather than read from global state.

**This module has no runtime dependency on ``review/``.** It is handed an
evidence pack and a draft, reads them, and returns its own
:class:`JudgeVerdict` — the same one-way relationship the Weight Agent has
with the rule engine. The type hints are quoted under ``TYPE_CHECKING`` and
the one constant it borrows is imported inside the function that needs it, so
the deterministic package can import this agent's models without a cycle.

What the agent can and cannot do is a property of its return type, not of its
prompt: :class:`JudgeVerdict` carries booleans, indices and short diagnostic
strings, so there is no field through which a model could introduce a number,
a signal value, or replacement wording.

Every failure — an unreachable provider, unreadable JSON, a verdict that
judges the wrong number of lines, the agent being switched off — returns an
unapproved verdict carrying ``parse_error`` rather than raising. The caller's
fallback is the deterministic template set, which is a good answer; failing
the batch instead would turn a degraded explanation into no explanation.
"""

from __future__ import annotations

import json
import time
from string import Template
from typing import TYPE_CHECKING

from maxxflow_core.errors import get_logger
from maxxflow_core.jsonutil import json_default
from maxxflow_core.ports import GenerationConfig, LLMProvider

from m3_production_delay.llm_agents.review_agent.config import (
    ReviewAgentConfig,
    get_review_agent_config,
)
from m3_production_delay.llm_agents.review_agent.exceptions import JudgeResponseError
from m3_production_delay.llm_agents.review_agent.models import (
    SKIPPED_VERDICT,
    JudgeLineVerdict,
    JudgeVerdict,
    coerce_diagnostics,
)
from m3_production_delay.llm_agents.review_agent.tracing import NOOP_TRACER, ReviewAgentTracer
from m3_production_delay.prompt import load_prompt_template

if TYPE_CHECKING:  # import-only: see the module docstring's one-way rule
    from m3_production_delay.review.schemas import EvidencePack, InsightDraft

log = get_logger("m3_production_delay.review_agent.resolver")

PROMPT_VERSION = "review-judge-v1"
AGENT_VERSION = "review-agent-v1"

#: Loaded once at import time from ``modules/m3_production_delay/prompt/`` —
#: the prompts' wording lives there, beside the Weight Agent's two, so it can
#: be reviewed and edited without touching this module.
_SYSTEM_PROMPT = load_prompt_template("review_judge_system.txt")
_USER_TEMPLATE = Template(load_prompt_template("review_judge_user.txt"))


def render_candidate_lines(draft: "InsightDraft") -> str:
    """The candidate lines as the judge sees them: index, the signal each
    claims, its scope, and the rendered text. The signal key is shown because
    the first two checks are about the line's *claim* — which signal it rests
    on — not only about its wording."""
    if not draft.why_lines:
        return "(none)"
    rendered = []
    for line in draft.why_lines:
        scope = f"operation {line.operation_id}" if line.operation_id else "job"
        rendered.append(
            f"[{line.index}] signal={line.signal_key} scope={scope}\n"
            f"      {line.headline}\n"
            f"      {line.detail}" + (f" ({line.delta})" if line.delta else "")
        )
    return "\n".join(rendered)


def build_user_prompt(pack: "EvidencePack", draft: "InsightDraft") -> str:
    """Evidence pack plus indexed candidate lines. ``allow_nan=False`` so a
    non-finite number can never reach the model as a bare ``Infinity``
    literal, which is not JSON it could be expected to read."""
    # Imported here, not at module scope: the fire baselines are the
    # deterministic package's single source of truth for what "fired" means,
    # and this agent must not depend on that package at import time (see the
    # module docstring). The prompt has to state the same thresholds the
    # composer and validators used, so it borrows them rather than restating
    # numbers that could drift.
    from m3_production_delay.review.evidence import FIRE_BASELINES

    evidence_json = json.dumps(
        pack.to_dict(), indent=2, default=json_default, allow_nan=False, sort_keys=True
    )
    baselines = "\n".join(f"  - {key}: > {value}" for key, value in sorted(FIRE_BASELINES.items()))
    return _USER_TEMPLATE.substitute(
        evidence_json=evidence_json,
        fire_baselines=baselines,
        candidate_lines=render_candidate_lines(draft),
    )


def strip_code_fence(raw_text: str) -> str:
    """Unwrap a markdown code fence around an otherwise-clean JSON response.

    The system prompt asks for "JSON only, no prose before or after it", and
    models still routinely answer with::

        ```json
        {"approved": true, ...}
        ```

    Verified against a real Foundry deployment: that is what comes back, and
    ``json.loads`` fails on it at character 0. Unwrapping the fence is not
    leniency about the contract — the content inside is exactly the contract —
    it just declines to fail over the packaging. Everything after this stays
    strict: prose outside a fence, or a fenced non-object, still raises.
    """
    text = raw_text.strip()
    if not text.startswith("```"):
        return text
    # Drop the opening fence line, which may carry a language tag (```json).
    _, _, remainder = text.partition("\n")
    closing = remainder.rfind("```")
    return (remainder[:closing] if closing != -1 else remainder).strip()


def load_verdict_object(raw_text: str) -> object:
    """Read ONE JSON document from the start of the response.

    ``json.loads`` requires the whole string to be that document, and a real
    model routinely appends a sentence of commentary after it — which fails as
    ``Extra data: line 37 column 1``. ``raw_decode`` parses the first complete
    value and simply does not consume the rest.

    Trailing commentary is therefore tolerated; a *leading* preamble is not.
    That asymmetry is deliberate rather than lazy: decoding from position zero
    is a well-defined operation, whereas hunting for the first ``{`` inside
    arbitrary prose is guessing where the document starts, and a wrong guess
    would silently judge on a fragment. Anything this cannot read falls back to
    the deterministic templates, which is a safe outcome rather than a broken
    one.
    """
    payload, _end = json.JSONDecoder().raw_decode(strip_code_fence(raw_text))
    return payload


def _parse_verdict(raw_text: str, line_count: int) -> JudgeVerdict:
    """Strict parse, in the ``profile_extractor._parse_response`` style:
    ``json.loads`` then a structural check of every field, raising
    :class:`JudgeResponseError` on anything unexpected."""
    try:
        payload = load_verdict_object(raw_text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise JudgeResponseError(f"invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise JudgeResponseError("schema_mismatch: response is not a JSON object")

    approved = payload.get("approved")
    if not isinstance(approved, bool):
        raise JudgeResponseError("schema_mismatch: 'approved' is not a boolean")

    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list):
        raise JudgeResponseError("schema_mismatch: 'lines' is not an array")

    verdicts: list[JudgeLineVerdict] = []
    seen: set[int] = set()
    for entry in raw_lines:
        if not isinstance(entry, dict):
            raise JudgeResponseError("schema_mismatch: a line verdict is not an object")
        index = entry.get("line_index")
        supported = entry.get("supported")
        if isinstance(index, bool) or not isinstance(index, int):
            raise JudgeResponseError("schema_mismatch: 'line_index' is not an integer")
        if not 0 <= index < line_count:
            raise JudgeResponseError(
                f"out_of_range: line_index {index} is not one of the {line_count} candidate lines"
            )
        if index in seen:
            raise JudgeResponseError(f"duplicate_line_index: {index}")
        if not isinstance(supported, bool):
            raise JudgeResponseError("schema_mismatch: 'supported' is not a boolean")
        seen.add(index)
        verdicts.append(JudgeLineVerdict.from_dict(entry))

    if len(seen) != line_count:
        raise JudgeResponseError(
            f"incomplete_verdict: {len(seen)} of {line_count} candidate lines were judged"
        )

    unsupported = [v for v in verdicts if not v.supported]
    omitted = coerce_diagnostics(payload.get("omitted_signals"))
    claims = coerce_diagnostics(payload.get("unsupported_claims"))
    # A verdict that approves while marking a line unsupported (or naming an
    # omission) contradicts itself. The conservative reading wins: the dissent
    # is what gets acted on, never the blanket approval.
    return JudgeVerdict(
        approved=approved and not unsupported and not omitted,
        lines=tuple(verdicts),
        unsupported_claims=claims,
        omitted_signals=omitted,
    )


class ReviewAgent:
    """Wraps exactly one :class:`LLMProvider` call per judging round.

    The round budget itself (one call, plus at most one re-judge after
    dropping unsupported lines) belongs to ``review/pipeline.py``, because
    only the pipeline can change the question between rounds — this agent
    never rewrites a line, so retrying an unchanged draft here could not
    change the answer. Both read the budget from the same
    :class:`ReviewAgentConfig`.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider | None = None,
        config: ReviewAgentConfig | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._config = config or get_review_agent_config()

    @property
    def config(self) -> ReviewAgentConfig:
        return self._config

    def _resolve_llm_provider(self) -> LLMProvider:
        """Injected provider wins; otherwise the platform factory decides from
        ``LLM_PROVIDER``. Resolved lazily so constructing an agent never
        depends on a provider being configured — a batch that judges nothing
        must not fail at import."""
        if self._llm_provider is not None:
            return self._llm_provider
        from maxxflow_providers import get_llm_provider

        self._llm_provider = get_llm_provider()
        return self._llm_provider

    def judge(
        self,
        pack: "EvidencePack",
        draft: "InsightDraft",
        *,
        tracer: ReviewAgentTracer = NOOP_TRACER,
    ) -> JudgeVerdict:
        """Judge one draft against its evidence.

        Returns an approved, ``skipped`` verdict without calling anything when
        there are no candidate lines — there is nothing to judge, and a call
        would only invite the model to invent an objection.
        """
        if not draft.why_lines:
            tracer.trace_decision("JUDGE", "skipped_no_lines")
            return SKIPPED_VERDICT

        if not self._config.llm_enabled:
            # Switched off behaves exactly like an unreachable judge: the
            # deterministic lines still ship, marked as unjudged.
            tracer.trace_decision("JUDGE", "skipped_llm_disabled")
            return JudgeVerdict(approved=False, skipped=True, parse_error="llm_disabled")

        generation_config = GenerationConfig(
            temperature=self._config.temperature, seed=self._config.seed
        )
        user_prompt = build_user_prompt(pack, draft)
        provider = self._resolve_llm_provider()
        provider_name = getattr(provider, "name", type(provider).__name__)

        tracer.trace_stage(
            "LLM CALL",
            stage="insight_review",
            provider=type(provider).__name__,
            prompt_version=PROMPT_VERSION,
            candidate_lines=len(draft.why_lines),
            generation_temperature=generation_config.temperature,
            generation_seed=generation_config.seed,
        )
        tracer.trace_prompt("JUDGE", system=_SYSTEM_PROMPT, user=user_prompt)

        started = time.perf_counter()
        try:
            raw_text = provider.generate(
                f"{_SYSTEM_PROMPT}\n\n{user_prompt}",
                max_tokens=self._config.max_tokens,
                generation_config=generation_config,
            )
        except Exception as exc:  # provider timeout / transport / auth failure
            log.warning("m3_review judge call failed job_id=%s: %s", pack.job_id, exc)
            tracer.trace_decision("JUDGE RESPONSE", "provider_error", error=str(exc))
            verdict = JudgeVerdict(approved=False, parse_error=f"provider_error: {exc}")
            tracer.trace_final_summary(pack.job_id, verdict)
            return verdict

        latency_ms = int((time.perf_counter() - started) * 1000)
        raw_text = raw_text or ""
        # Sizes and latency, never the bodies: the evidence pack carries job
        # references and vendor names, and the tracer above owns the single
        # opt-in, content-gated path for the text itself.
        log.info(
            "m3_review judge_call job_id=%s provider=%s prompt_version=%s temperature=%s "
            "seed=%s prompt_chars=%d response_chars=%d latency_ms=%d",
            pack.job_id,
            provider_name,
            PROMPT_VERSION,
            generation_config.temperature,
            generation_config.seed,
            len(_SYSTEM_PROMPT) + len(user_prompt),
            len(raw_text),
            latency_ms,
        )
        tracer.trace_stage("JUDGE RESPONSE", response_length=len(raw_text), latency_ms=latency_ms)
        tracer.trace_response("JUDGE", raw_text)

        try:
            verdict = _parse_verdict(raw_text, len(draft.why_lines))
        except JudgeResponseError as exc:
            log.warning("m3_review judge response rejected job_id=%s: %s", pack.job_id, exc)
            tracer.trace_decision("JUDGE VALIDATION", "rejected", parse_error=str(exc))
            verdict = JudgeVerdict(approved=False, parse_error=str(exc))
        else:
            tracer.trace_decision(
                "JUDGE VALIDATION",
                "accepted",
                approved=verdict.approved,
                unsupported_line_indices=list(verdict.unsupported_indices),
            )
            log.info(
                "m3_review judged job_id=%s lines=%d approved=%s unsupported=%d",
                pack.job_id,
                len(draft.why_lines),
                verdict.approved,
                len(verdict.unsupported_indices),
            )

        tracer.trace_final_summary(pack.job_id, verdict)
        return verdict
