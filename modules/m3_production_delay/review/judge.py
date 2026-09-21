"""The LLM-as-a-judge step: one call, asked only whether each already-composed
line is supported by the evidence.

The call is made through a plain ``llm(system_prompt, user_prompt) -> str``
callable so the pipeline never depends on a provider type. :func:`build_llm`
builds one over ``maxxflow_providers.get_llm_provider()``, the same port the
Weight Agent's two calls use. The platform's :class:`LLMProvider` port is a
single text-in/text-out ``generate()``, so the two prompt halves are
concatenated — the split exists for the reader and for a future adapter that
does have a role-aware API, not because the wire format has one today.

Parsing follows ``llm_agents/weight_agent/profile_extractor.py``: ``json.loads``
then a strict structural check, with any failure producing an explicit
``approved=False`` verdict instead of an exception. An unreadable answer must
never be mistaken for assent, and a transport failure must never be able to
publish an unjudged insight.

**Under ``LLM_PROVIDER=stub`` — the default, and what CI runs — every judge
call returns prose, not JSON.** That parses as ``approved=False`` with a
``parse_error``, so a stub-backed run always ends at
``status="fallback_template"``: the deterministic template lines are
published, marked as not judged. That is the intended degraded behaviour, not
a misconfiguration.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from string import Template

from maxxflow_core.errors import get_logger
from maxxflow_core.jsonutil import json_default
from maxxflow_core.ports import GenerationConfig, LLMProvider

from m3_production_delay.prompt import load_prompt_template
from m3_production_delay.review.evidence import FIRE_BASELINES
from m3_production_delay.review.schemas import (
    EvidencePack,
    InsightDraft,
    JudgeLineVerdict,
    JudgeVerdict,
)

log = get_logger("m3_production_delay.review.judge")

PROMPT_VERSION = "review-judge-v1"

#: Loaded once at import, like every other prompt in this module — the wording
#: lives in ``prompt/`` so it can be reviewed without touching code.
_SYSTEM_PROMPT = load_prompt_template("review_judge_system.txt")
_USER_TEMPLATE = Template(load_prompt_template("review_judge_user.txt"))

#: Fixed so a retry reuses an identical request. See ``ports.GenerationConfig``
#: for why this narrows variance rather than promising byte-identical output;
#: the verdict is validated structurally either way.
_GENERATION_CONFIG = GenerationConfig(temperature=0.0, seed=0)

#: Enough for one verdict object over a realistic job (a few lines, short
#: refs). The prompt forbids prose, so a response near this ceiling is itself
#: a sign the model ignored the contract — and will fail the schema check.
MAX_TOKENS = 900

#: A judge callable: (system_prompt, user_prompt) -> raw response text.
LLMCallable = Callable[[str, str], str]


def build_llm(provider: LLMProvider | None = None) -> LLMCallable:
    """Adapt an :class:`LLMProvider` to the callable the judge takes.

    Logs latency and response size for every call. It deliberately does not
    log the prompt or the response body: the evidence pack contains vendor
    names and job references, and the Weight Agent's tracer already owns the
    opt-in, content-gated path for anyone who needs to see prompt text.
    """
    if provider is None:
        from maxxflow_providers import get_llm_provider

        provider = get_llm_provider()

    def llm(system_prompt: str, user_prompt: str) -> str:
        started = time.perf_counter()
        response = provider.generate(
            f"{system_prompt}\n\n{user_prompt}",
            max_tokens=MAX_TOKENS,
            generation_config=_GENERATION_CONFIG,
        )
        log.info(
            "m3_review judge_call provider=%s prompt_version=%s temperature=%s seed=%s "
            "prompt_chars=%d response_chars=%d latency_ms=%d",
            getattr(provider, "name", type(provider).__name__),
            PROMPT_VERSION,
            _GENERATION_CONFIG.temperature,
            _GENERATION_CONFIG.seed,
            len(system_prompt) + len(user_prompt),
            len(response or ""),
            int((time.perf_counter() - started) * 1000),
        )
        return response

    return llm


def render_candidate_lines(draft: InsightDraft) -> str:
    """The candidate lines as the judge sees them: index, scope, the signal
    each claims, and the rendered text. The signal key is shown because check
    1 and 2 are about the line's *claim*, which is the signal it rests on."""
    if not draft.why_lines:
        return "(none)"
    rendered = []
    for line in draft.why_lines:
        scope = (
            f"operation {line.operation_id}" if line.operation_id else "job"
        )
        rendered.append(
            f"[{line.index}] signal={line.signal_key} scope={scope}\n"
            f"      {line.headline}\n"
            f"      {line.detail}" + (f" ({line.delta})" if line.delta else "")
        )
    return "\n".join(rendered)


def build_user_prompt(pack: EvidencePack, draft: InsightDraft) -> str:
    evidence_json = json.dumps(
        pack.to_dict(), indent=2, default=json_default, allow_nan=False, sort_keys=True
    )
    baselines = "\n".join(
        f"  - {key}: > {value}" for key, value in sorted(FIRE_BASELINES.items())
    )
    return _USER_TEMPLATE.substitute(
        evidence_json=evidence_json,
        fire_baselines=baselines,
        candidate_lines=render_candidate_lines(draft),
    )


def _parse_verdict(raw_text: str, line_count: int) -> JudgeVerdict:
    """Strict parse. Every failure path returns ``approved=False`` with the
    reason recorded, so a caller can never confuse "could not read the
    verdict" with "the verdict was yes"."""
    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError) as exc:
        return JudgeVerdict(approved=False, parse_error=f"invalid_json: {exc}")
    if not isinstance(payload, dict):
        return JudgeVerdict(
            approved=False, parse_error="schema_mismatch: response is not a JSON object"
        )

    approved = payload.get("approved")
    if not isinstance(approved, bool):
        return JudgeVerdict(
            approved=False, parse_error="schema_mismatch: 'approved' is not a boolean"
        )

    raw_lines = payload.get("lines")
    if not isinstance(raw_lines, list):
        return JudgeVerdict(
            approved=False, parse_error="schema_mismatch: 'lines' is not an array"
        )

    verdicts: list[JudgeLineVerdict] = []
    seen: set[int] = set()
    for entry in raw_lines:
        if not isinstance(entry, dict):
            return JudgeVerdict(
                approved=False, parse_error="schema_mismatch: a line verdict is not an object"
            )
        index = entry.get("line_index")
        supported = entry.get("supported")
        if isinstance(index, bool) or not isinstance(index, int):
            return JudgeVerdict(
                approved=False, parse_error="schema_mismatch: 'line_index' is not an integer"
            )
        if not 0 <= index < line_count:
            return JudgeVerdict(
                approved=False,
                parse_error=f"out_of_range: line_index {index} is not one of the {line_count} "
                "candidate lines",
            )
        if index in seen:
            return JudgeVerdict(
                approved=False, parse_error=f"duplicate_line_index: {index}"
            )
        if not isinstance(supported, bool):
            return JudgeVerdict(
                approved=False, parse_error="schema_mismatch: 'supported' is not a boolean"
            )
        seen.add(index)
        verdicts.append(
            JudgeLineVerdict(
                line_index=index,
                supported=supported,
                evidence_ref=_short_text(entry.get("evidence_ref")),
                issue=_short_text(entry.get("issue")),
            )
        )

    if len(seen) != line_count:
        return JudgeVerdict(
            approved=False,
            parse_error=(
                f"incomplete_verdict: {len(seen)} of {line_count} candidate lines were judged"
            ),
        )

    unsupported = [v for v in verdicts if not v.supported]
    omitted = _string_tuple(payload.get("omitted_signals"))
    # A verdict that approves while marking a line unsupported (or naming an
    # omission) contradicts itself. The conservative reading wins: the
    # dissent is what gets acted on, never the blanket approval.
    consistent_approval = approved and not unsupported and not omitted

    return JudgeVerdict(
        approved=consistent_approval,
        lines=tuple(verdicts),
        unsupported_claims=_string_tuple(payload.get("unsupported_claims")),
        omitted_signals=omitted,
    )


def _short_text(value: object, limit: int = 300) -> str | None:
    if value is None:
        return None
    return str(value)[:limit]


def _string_tuple(value: object, limit: int = 20) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item)[:300] for item in value if isinstance(item, (str, int, float)))[:limit]


def judge_draft(pack: EvidencePack, draft: InsightDraft, llm: LLMCallable) -> JudgeVerdict:
    """One judging call for one draft.

    A draft with no candidate lines is not sent: there is nothing to judge,
    and a call would only invite the model to invent an objection. That case
    returns an approved, ``skipped=True`` verdict so "nobody asked" stays
    distinguishable from "the judge agreed".

    A provider that raises (timeout, transport, auth) is caught and reported
    as an unapproved verdict — the pipeline's fallback is the deterministic
    template, which is strictly better than failing the whole batch.
    """
    if not draft.why_lines:
        return JudgeVerdict(approved=True, skipped=True)

    user_prompt = build_user_prompt(pack, draft)
    try:
        raw_text = llm(_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # provider timeout / transport / auth failure
        log.warning("m3_review judge call failed job_id=%s: %s", pack.job_id, exc)
        return JudgeVerdict(approved=False, parse_error=f"provider_error: {exc}")

    verdict = _parse_verdict(raw_text or "", len(draft.why_lines))
    log.info(
        "m3_review judged job_id=%s lines=%d approved=%s unsupported=%d parse_error=%s",
        pack.job_id,
        len(draft.why_lines),
        verdict.approved,
        len(verdict.unsupported_indices),
        verdict.parse_error,
    )
    return verdict
