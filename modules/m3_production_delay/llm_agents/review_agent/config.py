"""Single source of truth for the Review Agent's judging parameters.

Mirrors ``weight_agent/config.py``: a cached singleton built from typed
constants and validated once in ``__post_init__``, not in a separate function
a caller could forget to call. ``get_review_agent_config.cache_clear()``
reloads in tests.

Anything that needs a judging parameter — the agent itself, and
``review/pipeline.py`` for the attempt budget — imports
:func:`get_review_agent_config` rather than holding its own copy, so the
retry budget the pipeline spends and the one the agent documents cannot
drift apart.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from maxxflow_core.settings import get_settings

from m3_production_delay.llm_agents.review_agent.exceptions import ReviewConfigError

#: Enough for one verdict object over a realistic job — a few lines, each with
#: a short reference and issue. The prompt forbids prose, so a response that
#: approaches this ceiling has already ignored the contract and will fail the
#: schema check anyway.
DEFAULT_MAX_TOKENS = 900

#: Total judging calls per job: one, plus at most one re-judge after dropping
#: the lines the first verdict would not support.
#:
#: Deliberately NOT the Weight Agent's "retry the identical request once"
#: pattern. An identical retry is worth it there because its fallback is a
#: generic prior; here the fallback is the deterministic template set, which
#: has already passed every validator and is a genuinely good answer. Spending
#: a second call to re-ask an unchanged question would buy a small chance of
#: fixing transient bad JSON at the cost of doubling the bill on every
#: unreachable-provider run. The retry we do spend changes the question.
DEFAULT_MAX_ATTEMPTS = 2

#: Fixed so a re-judge differs from the first call only in which lines it
#: carries. See ``ports.GenerationConfig`` for why this narrows variance
#: rather than promising byte-identical output — the verdict is validated
#: structurally either way.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 0


@dataclass(frozen=True)
class ReviewAgentConfig:
    """Self-validating: constructing one directly (not just via
    :func:`build_config`) still enforces every invariant."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    temperature: float = DEFAULT_TEMPERATURE
    seed: int | None = DEFAULT_SEED
    llm_enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ReviewConfigError(f"max_tokens={self.max_tokens} must be >= 1")
        if self.max_attempts < 1:
            raise ReviewConfigError(
                f"max_attempts={self.max_attempts} must be >= 1 — a zero budget would "
                "publish every insight unjudged while still paying for the pipeline"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ReviewConfigError(
                f"temperature={self.temperature} must be within [0, 2]"
            )
        if not isinstance(self.llm_enabled, bool):
            raise ReviewConfigError("llm_enabled must be a bool")


def build_config(
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int | None = DEFAULT_SEED,
    llm_enabled: bool | None = None,
) -> ReviewAgentConfig:
    """Construct a config. Defaults reproduce the shipped v1 values; tests
    pass overrides. Validation lives on ``ReviewAgentConfig.__post_init__`` —
    this function only supplies defaults and the ``Settings`` lookup for
    ``llm_enabled``, which module code must never read from the environment
    itself."""
    if llm_enabled is None:
        llm_enabled = get_settings().m3_review_llm_enabled
    return ReviewAgentConfig(
        max_tokens=max_tokens,
        max_attempts=max_attempts,
        temperature=temperature,
        seed=seed,
        llm_enabled=llm_enabled,
    )


@functools.lru_cache(maxsize=1)
def get_review_agent_config() -> ReviewAgentConfig:
    """Cached singleton — the shipped v1 config. Call
    ``get_review_agent_config.cache_clear()`` in tests to reload."""
    return build_config()
