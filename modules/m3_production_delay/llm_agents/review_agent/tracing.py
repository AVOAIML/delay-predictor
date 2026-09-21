"""Optional, opt-in execution trace for one judging call, for local
development and debugging only.

Same two-switch contract as ``weight_agent/tracing.py``
(``Settings.m3_review_trace_enabled`` / ``m3_review_trace_include_content``,
both default ``False``), and a separate logger for the same reason: it never
propagates to the root logger, so a handler attached there for a real
destination can never receive raw prompt or response content just because
tracing was switched on.

* disabled (default): every method is a no-op — a boolean check and nothing else.
* enabled, content off: stage names, decisions and safe metadata (line counts,
  response length, parse outcome, which line indices were refused).
* enabled, content on: the above plus the raw prompt and raw response, through
  :meth:`trace_prompt` / :meth:`trace_response`. That is the only path through
  which the evidence pack's own text — job references, vendor names — is ever
  logged by this package.

A deliberately separate tracer rather than an import of the Weight Agent's:
the two agents have different stages and different content to gate, and
coupling them would mean one agent's trace flag silently governing the
other's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from m3_production_delay.llm_agents.review_agent.models import JudgeVerdict

_TRACE_LOGGER_NAME = "m3_production_delay.review_agent.trace"


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


class ReviewAgentTracer:
    """One instance per :meth:`ReviewAgent.judge` call — holds the trace id
    and the two enablement flags, passed explicitly down the call chain
    rather than through implicit global state."""

    def __init__(self, *, trace_id: str, enabled: bool, include_content: bool) -> None:
        self.trace_id = trace_id
        self.enabled = enabled
        self.include_content = include_content and enabled
        self._log = _build_trace_logger()

    def trace_stage(self, label: str, **fields: object) -> None:
        if not self.enabled:
            return
        self._emit(label, fields)

    def trace_decision(self, label: str, decision: str, **fields: object) -> None:
        if not self.enabled:
            return
        self._emit(label, {"decision": decision, **fields})

    def trace_prompt(self, label: str, *, system: str, user: str) -> None:
        """Content-gated. Traces the two halves separately, as they are
        written — never a fabricated single blob, even though the platform's
        :class:`LLMProvider` port concatenates them on the way out."""
        if not self.enabled or not self.include_content:
            return
        self._log.info(
            "\n".join(
                [
                    f"----- {label} PROMPT START -----",
                    "SYSTEM:",
                    system,
                    "USER:",
                    user,
                    f"----- {label} PROMPT END -----",
                ]
            )
        )

    def trace_response(self, label: str, raw_response: str) -> None:
        """Content-gated. Response metadata (length, parse outcome) is traced
        separately by the caller so it stays visible with content off."""
        if not self.enabled or not self.include_content:
            return
        self._log.info(
            "----- %s RESPONSE START -----\n%s\n----- %s RESPONSE END -----",
            label,
            raw_response,
            label,
        )

    def trace_final_summary(self, job_id: str, verdict: JudgeVerdict) -> None:
        if not self.enabled:
            return
        refused = list(verdict.unsupported_indices)
        self._log.info(
            "\n========== M3 REVIEW AGENT VERDICT ==========\n"
            f"trace_id: {self.trace_id}\n"
            f"job_id: {job_id}\n\n"
            f"approved: {str(verdict.approved).lower()}\n"
            f"skipped: {str(verdict.skipped).lower()}\n"
            f"lines_judged: {len(verdict.lines)}\n"
            f"unsupported_line_indices: {refused}\n"
            f"omitted_signals: {list(verdict.omitted_signals)}\n"
            f"parse_error: {verdict.parse_error}\n"
            "============================================="
        )

    def _emit(self, stage: str, fields: Mapping[str, object]) -> None:
        rendered = "\n".join(f"{key}={_format_field(value)}" for key, value in fields.items())
        header = f"[M3 REVIEW TRACE] {stage}\ntrace_id={self.trace_id}"
        self._log.info(f"{header}\n{rendered}" if rendered else header)


#: Shared, immutable, never mutated after construction — safe as a default
#: parameter value, so no call site needs an ``Optional`` + ``is None`` check.
NOOP_TRACER = ReviewAgentTracer(trace_id="disabled", enabled=False, include_content=False)
