"""WeightAgentTracer unit tests, plus a small number of resolver-level
integration tests proving the trace wiring behaves as specified: content
gating, the profile/adjustment prompt trust-boundary separation, and visible
fitted-path/fallback-path fields. Does not duplicate WeightAgent's own
statistical test suite (test_resolver.py) — only what tracing adds.
"""

from __future__ import annotations

from m3_production_delay.llm_agents.weight_agent.config import build_config
from m3_production_delay.llm_agents.weight_agent.exceptions import AllSignalsUnavailableError
from m3_production_delay.llm_agents.weight_agent.history_policy import compute_usable_floor
from m3_production_delay.llm_agents.weight_agent.models import SIGNAL_ORDER, FittedWeights
from m3_production_delay.llm_agents.weight_agent.resolver import WeightAgent
from m3_production_delay.llm_agents.weight_agent.tracing import NOOP_TRACER, WeightAgentTracer
import pytest

CONFIG = build_config()
ALL_AVAILABLE = {signal: True for signal in SIGNAL_ORDER}


class _RecordingSink:
    """Stands in for the real logging.Logger the tracer would otherwise use —
    avoids fighting stdlib logging propagation semantics (the trace logger
    deliberately sets propagate=False, so pytest's caplog cannot see it)."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: object, *args: object) -> None:
        self.messages.append(str(msg) % args if args else str(msg))

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


def _tracer(*, enabled: bool, include_content: bool) -> tuple[WeightAgentTracer, _RecordingSink]:
    tracer = WeightAgentTracer(trace_id="test-trace-id", enabled=enabled, include_content=include_content)
    sink = _RecordingSink()
    tracer._log = sink  # type: ignore[assignment]
    return tracer, sink


_PROFILE_JSON = (
    '{"profile": {"production_type": "make_to_order", "material_dependency": "high", '
    '"supplier_dependency": "high", "workforce_dependency": "low", "seasonality_level": "low"}}'
)
_ADJUSTMENT_JSON = (
    '{"adjustments_bp": {"time_overrun": 200, "operator_skill": -200, "seasonality": 0, '
    '"material_availability": 0, "supplier_reliability": 0}, "evidence": ["ok"]}'
)


class _TwoCallLLMProvider:
    name = "two-call-fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, max_tokens: int = 256, generation_config=None) -> str:
        self.calls += 1
        return _PROFILE_JSON if self.calls == 1 else _ADJUSTMENT_JSON


class _FixedFittedProvider:
    def __init__(self, fitted: FittedWeights) -> None:
        self._fitted = fitted

    def get(self, tenant_id: str) -> FittedWeights | None:
        return self._fitted


# --- disabled ------------------------------------------------------------


def test_disabled_tracer_emits_nothing_at_all():
    tracer, sink = _tracer(enabled=False, include_content=False)
    tracer.trace_stage("STAGE", a=1)
    tracer.trace_decision("STAGE", "decision", a=1)
    tracer.trace_weights("STAGE", weights_bp={"x": 1})
    tracer.trace_prompt("LABEL", prompt="raw prompt text")
    tracer.trace_response("LABEL", "raw response text")
    tracer.trace_summary("summary text")
    assert sink.messages == []


def test_noop_tracer_constant_is_disabled():
    assert NOOP_TRACER.enabled is False
    assert NOOP_TRACER.include_content is False


# --- enabled, content off -------------------------------------------------


def test_enabled_without_content_emits_stage_metadata_but_no_prompt_or_response_text():
    tracer, sink = _tracer(enabled=True, include_content=False)
    tracer.trace_stage("LLM CALL #1", stage="tenant_profile_extraction", provider="X")
    tracer.trace_prompt("PROFILE EXTRACTION", prompt="SECRET RAW TENANT PROSE")
    tracer.trace_response("PROFILE EXTRACTION", "SECRET RAW RESPONSE")

    assert any("LLM CALL #1" in m for m in sink.messages)
    assert not any("SECRET RAW TENANT PROSE" in m for m in sink.messages)
    assert not any("SECRET RAW RESPONSE" in m for m in sink.messages)
    assert not any("PROMPT START" in m for m in sink.messages)
    assert not any("RESPONSE START" in m for m in sink.messages)


# --- enabled, content on --------------------------------------------------


def test_enabled_with_content_emits_prompt_and_response_text():
    tracer, sink = _tracer(enabled=True, include_content=True)
    tracer.trace_prompt("PROFILE EXTRACTION", prompt="SECRET RAW TENANT PROSE")
    tracer.trace_response("PROFILE EXTRACTION", "SECRET RAW RESPONSE")

    assert any("SECRET RAW TENANT PROSE" in m for m in sink.messages)
    assert any("SECRET RAW RESPONSE" in m for m in sink.messages)
    assert any("PROFILE EXTRACTION PROMPT START" in m for m in sink.messages)
    assert any("PROFILE EXTRACTION RESPONSE START" in m for m in sink.messages)


def test_trace_id_appears_on_every_emitted_line():
    tracer, sink = _tracer(enabled=True, include_content=False)
    tracer.trace_stage("A", x=1)
    tracer.trace_decision("B", "decided", y=2)
    tracer.trace_weights("C", weights_bp={"time_overrun": 1000})
    assert sink.messages, "expected at least one message"
    assert all("trace_id=test-trace-id" in m for m in sink.messages)


# --- secrets are structurally never reachable -----------------------------


def test_messages_json_serializer_receives_no_authorization_like_field():
    # trace_stage/trace_decision only ever emit fields the caller explicitly
    # passes; nothing in this module ever forwards a raw headers mapping.
    tracer, sink = _tracer(enabled=True, include_content=True)
    tracer.trace_stage("TENANT CONTEXT", checking="request_header", result="found")
    assert not any("authorization" in m.lower() for m in sink.messages)
    assert not any("bearer" in m.lower() for m in sink.messages)


# --- resolver-level integration: two-call trust boundary ------------------


def test_adjustment_prompt_trace_never_contains_the_raw_tenant_description():
    tracer, sink = _tracer(enabled=True, include_content=True)
    marker = "UNIQUE-RAW-DESCRIPTION-MARKER-4711"
    agent = WeightAgent(llm_provider=_TwoCallLLMProvider())

    result = agent.resolve(
        "tenant-trace",
        availability=ALL_AVAILABLE,
        tenant_description=f"A manufacturer. {marker}",
        tracer=tracer,
    )

    assert result.source == "llm_adjusted_prior"
    profile_block = sink.text.split("WEIGHT ADJUSTMENT PROMPT START")[0]
    adjustment_block = sink.text.split("WEIGHT ADJUSTMENT PROMPT START")[1]
    assert marker in profile_block  # stage 1 legitimately receives it
    assert marker not in adjustment_block  # stage 2 must never see it


# --- resolver-level integration: fitted path fields -----------------------


def test_fitted_path_trace_shows_signal_set_n_floor_lambda_and_projection():
    fitted = FittedWeights(
        weights_bp={signal: 2000 for signal in SIGNAL_ORDER},
        signal_set=frozenset(SIGNAL_ORDER),
        delayed_event_count=500,
    )
    tracer, sink = _tracer(enabled=True, include_content=False)
    agent = WeightAgent(fitted_provider=_FixedFittedProvider(fitted))

    result = agent.resolve("tenant-fitted", availability=ALL_AVAILABLE, tracer=tracer)

    assert result.source in ("blended", "historically_fitted")
    expected_floor = compute_usable_floor(5, CONFIG.history_policy.events_per_parameter)
    assert any("FITTED ARTIFACT" in m for m in sink.messages)
    assert any(f"n_floor={expected_floor}" in m for m in sink.messages)
    assert any("lambda_bp=" in m for m in sink.messages)
    assert any("projection_applied=True" in m for m in sink.messages)


# --- resolver-level integration: fallback path shows its reason -----------


def test_fallback_path_trace_shows_the_reason_no_description_supplied():
    tracer, sink = _tracer(enabled=True, include_content=False)
    agent = WeightAgent()

    result = agent.resolve("tenant-fallback", availability=ALL_AVAILABLE, tracer=tracer)

    assert result.source == "prior"
    assert "no_tenant_description_supplied" in list(result.fallback_reasons)
    assert any("route=cold_start" in m for m in sink.messages)
    assert any("fitted_weights_available=False" in m for m in sink.messages)


def test_all_signals_unavailable_trace_shows_the_rejection_before_raising():
    tracer, sink = _tracer(enabled=True, include_content=False)
    agent = WeightAgent()

    with pytest.raises(AllSignalsUnavailableError):
        agent.resolve(
            "tenant-none",
            availability={signal: False for signal in SIGNAL_ORDER},
            tracer=tracer,
        )

    assert any("all_signals_unavailable" in m for m in sink.messages)
