"""Shared builders for the Section 3 review tests.

The rollup-shaped builders (``make_op``, ``make_operator``, ``make_component``,
``make_history_entry``, ``make_vendor``, ``make_po``) already exist in
``tests/rule_engine/test_m3_elements.py`` and are the established way to state
only what a test actually varies. They are loaded from there by path rather
than re-declared: a second copy would be free to drift from the rollup shape
the rule engine really produces, which is exactly the thing these tests exist
to check against.

Every test in this package is offline. The judge is always a real
:class:`ReviewAgent` driven by a scripted ``LLMProvider`` — a class returning
fixed text — never a mock. That is the same style the existing M3 tests use
for the Weight Agent's LLM calls (``orchestrator._DemoScriptedLLMProvider``),
and the only style that exercises the agent's real prompt assembly and
verdict parsing rather than asserting that a function was called.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from m3_production_delay.llm_agents.review_agent import ReviewAgent, build_config

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ELEMENTS_TESTS = _REPO_ROOT / "tests" / "rule_engine" / "test_m3_elements.py"


def _load_builders():
    spec = importlib.util.spec_from_file_location("m3_rule_engine_test_builders", _ELEMENTS_TESTS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_builders = _load_builders()

make_op = _builders.make_op
make_operator = _builders.make_operator
make_history_entry = _builders.make_history_entry
make_component = _builders.make_component
make_vendor = _builders.make_vendor
make_po = _builders.make_po

#: The tenant weight vector these tests score against. Deliberately NOT
#: summing to 1.0 — a resolved vector never does, because the Weight Agent's
#: `seasonality` share has no rule-engine equivalent and is dropped. Keeping
#: the realistic shape here is what makes the contribution-share assertions
#: meaningful.
WEIGHTS: dict[str, float] = {
    "time_overrun_ratio": 0.40,
    "operator_pace_ratio": 0.30,
    "material_shortfall_ratio": 0.15,
    "supplier_reliability": 0.05,
}
THRESHOLD = 1.0

FIXTURE_PATH = (
    _REPO_ROOT / "modules" / "m3_production_delay" / "review" / "fixtures" / "section1_job.json"
)


@pytest.fixture
def weights() -> dict[str, float]:
    return dict(WEIGHTS)


@pytest.fixture
def threshold() -> float:
    return THRESHOLD


@pytest.fixture
def section1_job() -> dict:
    """A real ``calculate_delay_elements_for_jobs()`` output, dumped from the
    entry point itself by ``review/fixtures/build_fixture.py``. Using the
    recorded output rather than a hand-written dict is what makes these tests
    fail if the Risk Engine's shape ever changes."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class ScriptedProvider:
    """A deterministic stand-in for a real :class:`LLMProvider`, built the
    same way ``orchestrator._DemoScriptedLLMProvider`` is: it answers the one
    call the agent makes with fixed text, so the agent's own prompt assembly,
    JSON parsing and schema checks all run for real.

    ``responder`` receives the combined prompt the port is given and returns
    the raw response text.
    """

    name = "scripted (test only)"

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[str] = []

    def generate(self, prompt: str, *, max_tokens: int = 256, generation_config=None) -> str:
        self.calls.append(prompt)
        return self._responder(prompt)


def agent_for(responder, **config_overrides) -> ReviewAgent:
    """A real ReviewAgent wired to a scripted provider."""
    return ReviewAgent(
        llm_provider=ScriptedProvider(responder),
        config=build_config(llm_enabled=True, **config_overrides),
    )


def _line_count(prompt: str) -> int:
    """How many candidate lines the agent put in front of the judge."""
    return prompt.count("] signal=")


def _approve_all(prompt: str) -> str:
    return json.dumps(
        {
            "approved": True,
            "lines": [
                {
                    "line_index": i,
                    "supported": True,
                    "evidence_ref": "operations[].signals[]",
                    "issue": None,
                }
                for i in range(_line_count(prompt))
            ],
            "unsupported_claims": [],
            "omitted_signals": [],
        }
    )


def _refuse(prompt: str, unsupported_indices) -> str:
    refuse = set(unsupported_indices) if unsupported_indices else {0}
    return json.dumps(
        {
            "approved": False,
            "lines": [
                {
                    "line_index": i,
                    "supported": i not in refuse,
                    "evidence_ref": None,
                    "issue": "not supported by the evidence" if i in refuse else None,
                }
                for i in range(_line_count(prompt))
            ],
            "unsupported_claims": ["line is not supported by the evidence"],
            "omitted_signals": [],
        }
    )


def approving_agent() -> ReviewAgent:
    """Supports every candidate line it is shown."""
    return agent_for(_approve_all)


def rejecting_agent(*unsupported_indices: int) -> ReviewAgent:
    """Refuses the given line indices every time it is called — including the
    re-judge, so the retry path terminates in ``fallback_template`` rather
    than looping."""
    return agent_for(lambda prompt: _refuse(prompt, unsupported_indices))


def rejecting_once_agent(*unsupported_indices: int) -> ReviewAgent:
    """Refuses on the first call and approves whatever survives on the
    second — the drop-and-re-judge path."""
    state = {"calls": 0}

    def responder(prompt: str) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            return _refuse(prompt, unsupported_indices)
        return _approve_all(prompt)

    return agent_for(responder)


def unparsable_agent() -> ReviewAgent:
    """What the default ``StubLLMProvider`` actually returns: prose, not JSON."""
    return agent_for(
        lambda prompt: "[stub-llm:0f1e2d3c] Suggested explanation for: You are a reviewer"
    )


def responding_agent(response: str) -> ReviewAgent:
    """Answers with one fixed response, whatever it is asked."""
    return agent_for(lambda prompt: response)


class _ExplodingProvider(ScriptedProvider):
    def __init__(self) -> None:
        super().__init__(lambda prompt: "")

    def generate(self, prompt: str, *, max_tokens: int = 256, generation_config=None) -> str:
        raise TimeoutError("provider timed out")


def exploding_agent() -> ReviewAgent:
    """A provider that cannot be reached at all."""
    return ReviewAgent(llm_provider=_ExplodingProvider(), config=build_config(llm_enabled=True))


def never_called_agent() -> ReviewAgent:
    """Fails the test if the agent is asked anything."""

    def responder(prompt: str) -> str:
        raise AssertionError("the judge must not be called on this path")

    return agent_for(responder)
