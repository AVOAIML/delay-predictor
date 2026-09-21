"""Shared builders for the Section 3 review tests.

The rollup-shaped builders (``make_op``, ``make_operator``, ``make_component``,
``make_history_entry``, ``make_vendor``, ``make_po``) already exist in
``tests/rule_engine/test_m3_elements.py`` and are the established way to state
only what a test actually varies. They are loaded from there by path rather
than re-declared: a second copy would be free to drift from the rollup shape
the rule engine really produces, which is exactly the thing these tests exist
to check against.

Every test in this package is offline. The judge is always a plain scripted
callable — a function returning fixed JSON — never a mock: the same style the
existing M3 tests use for the Weight Agent's LLM calls
(``orchestrator._DemoScriptedLLMProvider``), and the only style that exercises
the real parsing path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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


def approving_judge(system_prompt: str, user_prompt: str) -> str:
    """Scripted judge that supports every candidate line it is shown."""
    count = user_prompt.count("] signal=")
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
                for i in range(count)
            ],
            "unsupported_claims": [],
            "omitted_signals": [],
        }
    )


def rejecting_judge(*unsupported_indices: int):
    """Scripted judge that refuses the given line indices every time it is
    called — including the re-judge, so the retry path terminates in
    ``fallback_template`` rather than looping."""

    def judge(system_prompt: str, user_prompt: str) -> str:
        count = user_prompt.count("] signal=")
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
                    for i in range(count)
                ],
                "unsupported_claims": ["line is not supported by the evidence"],
                "omitted_signals": [],
            }
        )

    return judge


def rejecting_once_judge(*unsupported_indices: int):
    """Refuses on the first call and approves whatever survives on the second
    — the drop-and-re-judge path."""
    state = {"calls": 0}

    def judge(system_prompt: str, user_prompt: str) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            return rejecting_judge(*unsupported_indices)(system_prompt, user_prompt)
        return approving_judge(system_prompt, user_prompt)

    return judge


def unparsable_judge(system_prompt: str, user_prompt: str) -> str:
    """What the default ``StubLLMProvider`` actually returns: prose, not JSON."""
    return "[stub-llm:0f1e2d3c] Suggested explanation for: You are a reviewer"


def exploding_judge(system_prompt: str, user_prompt: str) -> str:
    raise TimeoutError("provider timed out")
