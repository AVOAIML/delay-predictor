"""Optional, opt-in execution trace for one Weight Agent resolution, for local
development/debugging only (spec: local trace mode). Completely separate from
the existing structured audit log (``resolver.WeightAgent._audit_log``) —
enabling or disabling this trace changes nothing about that log line, its
content, or its destination.

Two independent switches (``Settings.m3_weight_trace_enabled`` /
``m3_weight_trace_include_content``), both default ``False``:

* disabled (default): every method here is a no-op — zero log calls, zero
  formatting cost beyond a boolean check.
* enabled, content off: stage names, decisions, and safe metadata (lengths,
  counts, signal names, basis-point values) are emitted; no raw tenant
  description, no raw LLM prompt/response text.
* enabled, content on: the above, plus the raw prompt/response text emitted
  through :meth:`WeightAgentTracer.trace_prompt` / ``trace_response``. This
  is the only path through which that text is ever logged anywhere in this
  package.

The trace logger never propagates to the root logger (``propagate = False``)
so a handler attached there for a real destination (e.g. Application
Insights) can never receive raw prompt/response content just because tracing
was enabled — this logger only ever writes to its own local console handler.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER, WeightResolution, bp_to_percent

_TRACE_LOGGER_NAME = "m3_production_delay.weight_agent.trace"


def _build_trace_logger() -> logging.Logger:
    logger = logging.getLogger(_TRACE_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s TRACE :: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _format_field(value: object) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {v}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


class WeightAgentTracer:
    """One instance per ``WeightAgent.resolve()`` call — holds the trace id
    and the two enablement flags, and is passed explicitly down the existing
    call chain (``resolve`` -> ``TenantProfileExtractor.extract`` ->
    ``WeightAdjustmentGenerator.propose``, and ``TenantContextReader
    .resolve_description``) rather than through implicit global state.
    """

    def __init__(self, *, trace_id: str, enabled: bool, include_content: bool) -> None:
        self.trace_id = trace_id
        self.enabled = enabled
        self.include_content = include_content and enabled
        self._log = _build_trace_logger()

    def trace_stage(self, label: str, **fields: object) -> None:
        # `label`, not `stage` — several call sites pass a semantic
        # `stage="tenant_profile_extraction"` field alongside the block
        # label itself, which would otherwise collide with this parameter.
        if not self.enabled:
            return
        self._emit(label, fields)

    def trace_decision(self, label: str, decision: str, **fields: object) -> None:
        if not self.enabled:
            return
        self._emit(label, {"decision": decision, **fields})

    def trace_weights(self, label: str, **weight_fields: object) -> None:
        if not self.enabled:
            return
        self._emit(label, weight_fields)

    def trace_prompt(
        self,
        label: str,
        *,
        prompt: str | None = None,
        messages: list[tuple[str, str]] | None = None,
    ) -> None:
        """Content-gated. Traces the ACTUAL representation sent to the
        provider — separate ``messages`` if the provider abstraction uses
        them, never a fabricated combined prompt."""
        if not self.enabled or not self.include_content:
            return
        lines = [f"----- {label} PROMPT START -----"]
        if messages is not None:
            for role, text in messages:
                lines.append(f"{role.upper()}:")
                lines.append(text)
        elif prompt is not None:
            lines.append(prompt)
        lines.append(f"----- {label} PROMPT END -----")
        self._log.info("\n".join(lines))

    def trace_response(self, label: str, raw_response: str) -> None:
        """Content-gated. Metadata about the response (length) is a separate
        concern traced by the caller via ``trace_stage``/``trace_decision``
        so it is visible even when content tracing is off."""
        if not self.enabled or not self.include_content:
            return
        self._log.info(
            "----- %s RESPONSE START -----\n%s\n----- %s RESPONSE END -----",
            label,
            raw_response,
            label,
        )

    def trace_summary(self, text: str) -> None:
        if not self.enabled:
            return
        self._log.info(text)

    def trace_final_summary(self, result: WeightResolution) -> None:
        if not self.enabled:
            return
        percents = bp_to_percent(result.weights_bp)
        weight_lines = "\n".join(f"  {signal}: {percents[signal]:.2f}%" for signal in SIGNAL_ORDER)
        excluded = [e.signal for e in result.excluded_signals]
        text = (
            "\n========== M3 WEIGHT AGENT RESULT ==========\n"
            f"trace_id: {self.trace_id}\n"
            f"tenant_id: {result.tenant_id}\n\n"
            f"source: {result.source}\n"
            f"status: {result.status}\n\n"
            f"weights:\n{weight_lines}\n\n"
            f"available_signals:\n  {list(result.available_signals)}\n\n"
            f"excluded_signals:\n  {excluded}\n\n"
            f"requires_admin_approval: {str(result.requires_admin_approval).lower()}\n\n"
            f"fallback_reasons: {list(result.fallback_reasons)}\n\n"
            f"fitted_projection_applied: {str(result.fitted_projection_applied).lower()}\n"
            "============================================"
        )
        self._log.info(text)

    def _emit(self, stage: str, fields: Mapping[str, object]) -> None:
        rendered = "\n".join(f"{key}={_format_field(value)}" for key, value in fields.items())
        header = f"[M3 TRACE] {stage}\ntrace_id={self.trace_id}"
        self._log.info(f"{header}\n{rendered}" if rendered else header)


# Shared, immutable, never mutated after construction — safe as a default
# parameter value (same pattern as resolver._ZERO_ADJUSTMENT). Avoids every
# call site needing an `Optional[WeightAgentTracer]` + `is None` check.
NOOP_TRACER = WeightAgentTracer(trace_id="disabled", enabled=False, include_content=False)
