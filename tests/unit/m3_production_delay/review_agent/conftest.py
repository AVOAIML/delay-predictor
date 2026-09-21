"""Fixtures for the Review Agent's own tests.

The agent is handed an evidence pack and a draft and does not care where they
came from, so these build both from the recorded Section 1 fixture through the
real deterministic path — the same objects the pipeline would hand it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3_production_delay.review.composer import compose
from m3_production_delay.review.evidence import build_evidence

_REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_PATH = (
    _REPO_ROOT / "modules" / "m3_production_delay" / "review" / "fixtures" / "section1_job.json"
)

WEIGHTS: dict[str, float] = {
    "time_overrun_ratio": 0.40,
    "operator_pace_ratio": 0.30,
    "material_shortfall_ratio": 0.15,
    "supplier_reliability": 0.05,
}
THRESHOLD = 1.0


class ScriptedProvider:
    """Deterministic stand-in for a real ``LLMProvider``, in the same style as
    ``orchestrator._DemoScriptedLLMProvider`` — a class returning fixed text,
    never a mock, so the agent's own prompt assembly and parsing run for
    real."""

    name = "scripted (test only)"

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[str] = []
        self.max_tokens: list[int] = []
        self.generation_configs: list = []

    def generate(self, prompt: str, *, max_tokens: int = 256, generation_config=None) -> str:
        self.calls.append(prompt)
        self.max_tokens.append(max_tokens)
        self.generation_configs.append(generation_config)
        return self.response


@pytest.fixture
def pack():
    job = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return build_evidence(job, WEIGHTS, THRESHOLD)


@pytest.fixture
def draft(pack):
    return compose(pack)
