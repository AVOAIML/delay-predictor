"""Session-wide guards for the test suite.

The one rule this enforces: **tests never call a paid API.** That is a stated
invariant of the module (CLAUDE.md: "LLM/embedding calls are external APIs
behind provider ports with deterministic local stubs — tests never call a paid
API"), but nothing was enforcing it, and it stopped holding the moment a
developer put real credentials in `.env.local`.

`maxxflow_core.settings` reads that file, so any test that constructs a
provider from configuration — `ReviewAgent()` with no injected provider, say —
silently starts talking to a live deployment: slow, billable, non-deterministic,
and capable of failing a build because someone's quota ran out. Pinning the
provider to the stub for the whole session removes the possibility rather than
relying on every test to remember.

Tests that need a specific provider still set one explicitly with monkeypatch
(see `tests/unit/test_llm_providers.py`), which takes effect per-test on top of
this. The opt-in judge evaluation is the single deliberate exception.
"""

from __future__ import annotations

import os

import pytest

#: The one place a real provider is wanted: the judge evaluation script, which
#: exists to score a live model and is skipped unless this is set.
EVAL_FLAG = "M3_REVIEW_JUDGE_EVAL"

#: Cleared so a provider cannot be reached even if something overrides
#: LLM_PROVIDER mid-test.
_CREDENTIAL_VARS = (
    # Azure storage too: no test may write a snapshot to a real container.
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_ACCOUNT",
    "AZURE_AI_API_KEY",
    "AZURE_AI_API_BASE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_KEY",
)


@pytest.fixture(scope="session", autouse=True)
def _no_paid_api_calls() -> None:
    """Pin every provider to its deterministic stub for the whole session."""
    if os.environ.get(EVAL_FLAG):
        return  # the judge evaluation is explicitly asking for a real model

    from maxxflow_core.settings import get_settings

    os.environ["LLM_PROVIDER"] = "stub"
    os.environ["EMBEDDING_PROVIDER"] = "stub"
    os.environ["LLM_MODEL_ID"] = ""
    for name in _CREDENTIAL_VARS:
        # Set to empty rather than popped. pydantic-settings reads `.env.local`
        # from disk regardless of what is in os.environ, and a real environment
        # variable is the only thing that outranks it — deleting one just lets
        # the file's value through again.
        os.environ[name] = ""

    # Settings is an lru_cache singleton, and something may already have built
    # one from .env.local before this fixture ran.
    get_settings.cache_clear()
    try:
        from m3_production_delay.llm_agents.review_agent.config import (
            get_review_agent_config,
        )
        from m3_production_delay.llm_agents.weight_agent.config import (
            get_weight_agent_config,
        )

        get_review_agent_config.cache_clear()
        get_weight_agent_config.cache_clear()
    except ImportError:  # pragma: no cover - module layout changed
        pass
