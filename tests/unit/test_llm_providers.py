"""Azure AI Foundry LLM adapter — everything checkable without a network call.

The adapter itself is ``pragma: no cover`` for its request path, so what is
worth testing is the part that silently ruins a deployment: which credentials
it reads, how it normalises the endpoint, and that a missing or
wrong-data-plane configuration fails loudly instead of producing a client that
404s on first use.
"""

from __future__ import annotations

import pytest

from maxxflow_core.settings import get_settings
from maxxflow_providers.factory import _LLM, get_llm_provider
from maxxflow_providers.llm import (
    AzureAIFoundryLLMProvider,
    AzureOpenAILLMProvider,
    StubLLMProvider,
)


@pytest.fixture
def settings_env(monkeypatch):
    """Set provider settings through the environment, the way a profile does,
    and drop the Settings cache either side so no test leaks into another."""

    def apply(**values: str):
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        return get_settings()

    get_settings.cache_clear()
    yield apply
    get_settings.cache_clear()


# ─── factory routing ─────────────────────────────────────────────────────────


def test_every_provider_name_routes_to_an_adapter():
    assert _LLM["stub"] is StubLLMProvider
    assert _LLM["azure_ai"] is AzureAIFoundryLLMProvider
    # Azure OpenAI and plain OpenAI share a client; Foundry does not.
    assert _LLM["azure_openai"] is AzureOpenAILLMProvider
    assert _LLM["openai"] is AzureOpenAILLMProvider
    assert _LLM["azure_ai"] is not _LLM["azure_openai"]


def test_the_default_provider_is_the_offline_stub(settings_env):
    settings_env(LLM_PROVIDER="stub")
    assert isinstance(get_llm_provider(), StubLLMProvider)


# ─── endpoint normalisation ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        # A bare resource endpoint gets Foundry's inference route appended.
        (
            "https://res.services.ai.azure.com",
            "https://res.services.ai.azure.com/models",
        ),
        (
            "https://res.services.ai.azure.com/",
            "https://res.services.ai.azure.com/models",
        ),
        # Already complete: passed through untouched, not doubled up.
        (
            "https://res.services.ai.azure.com/models",
            "https://res.services.ai.azure.com/models",
        ),
        # An operator who pasted a different route is not second-guessed.
        (
            "https://res.services.ai.azure.com/openai/v1",
            "https://res.services.ai.azure.com/openai/v1",
        ),
    ],
)
def test_the_endpoint_is_normalised_to_the_inference_route(configured, expected):
    assert AzureAIFoundryLLMProvider.resolve_base_url(configured) == expected


# ─── failing loudly ──────────────────────────────────────────────────────────


def test_a_missing_deployment_id_is_refused(settings_env):
    settings_env(
        LLM_PROVIDER="azure_ai",
        LLM_MODEL_ID="",
        AZURE_AI_API_KEY="k",
        AZURE_AI_API_BASE="https://res.services.ai.azure.com",
    )
    # Verify-before-building: the deployment name is never defaulted, because
    # a guessed one fails at call time with a much less obvious error.
    with pytest.raises(NotImplementedError, match="DEPLOYMENT name"):
        get_llm_provider()


def test_missing_foundry_credentials_are_refused(settings_env):
    settings_env(
        LLM_PROVIDER="azure_ai",
        LLM_MODEL_ID="some-deployment",
        AZURE_AI_API_KEY="",
        AZURE_AI_API_BASE="",
    )
    with pytest.raises(NotImplementedError, match="AZURE_AI_API_KEY and AZURE_AI_API_BASE"):
        get_llm_provider()


def test_azure_openai_credentials_do_not_satisfy_the_foundry_adapter(settings_env):
    """The mistake this adapter exists to prevent: Foundry and Azure OpenAI
    are different data planes, so the OpenAI-side credentials cannot serve a
    Foundry deployment however complete they look."""
    settings_env(
        LLM_PROVIDER="azure_ai",
        LLM_MODEL_ID="some-deployment",
        AZURE_AI_API_KEY="",
        AZURE_AI_API_BASE="",
        AZURE_OPENAI_API_KEY="k",
        AZURE_OPENAI_ENDPOINT="https://res.openai.azure.com",
    )
    with pytest.raises(NotImplementedError, match="different data plane"):
        get_llm_provider()


# ─── settings surface ────────────────────────────────────────────────────────


def test_foundry_settings_are_read_from_their_own_aliases(settings_env):
    settings = settings_env(
        AZURE_AI_API_KEY="secret-key",
        AZURE_AI_API_BASE="https://res.services.ai.azure.com",
    )
    assert settings.azure_ai_api_base == "https://res.services.ai.azure.com"
    assert settings.azure_ai_api_key.get_secret_value() == "secret-key"
    # SecretStr, so an accidental log or repr cannot leak it.
    assert "secret-key" not in repr(settings.azure_ai_api_key)


def test_review_agent_settings_default_safely(settings_env):
    settings = settings_env(LLM_PROVIDER="stub")
    assert settings.m3_review_llm_enabled is True
    # Tracing off by default, and content tracing gated separately — the only
    # path through which raw prompt/response text is ever emitted.
    assert settings.m3_review_trace_enabled is False
    assert settings.m3_review_trace_include_content is False
