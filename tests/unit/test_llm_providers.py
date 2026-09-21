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


# ─── the Anthropic Messages surface ──────────────────────────────────────────
#
# Verified against a real Foundry staging resource: the /anthropic path answers
# POST /v1/messages on an `x-api-key` header, and rejects an OpenAI-style
# `api-key` with 401. These lock in the request shaping and response parsing
# that finding produced, without a network call.


@pytest.fixture
def foundry(settings_env):
    def build(base: str, model: str = "claude-sonnet-4-6"):
        settings_env(
            LLM_PROVIDER="azure_ai",
            LLM_MODEL_ID=model,
            AZURE_AI_API_KEY="test-key",
            AZURE_AI_API_BASE=base,
        )
        return get_llm_provider()

    return build


def test_the_path_selects_the_protocol(foundry):
    anthropic = foundry("https://res.services.ai.azure.com/anthropic")
    url, headers, body = anthropic.build_request(
        "hi", max_tokens=900, temperature=0.0, seed=0
    )
    assert url.endswith("/anthropic/v1/messages")
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == AzureAIFoundryLLMProvider.ANTHROPIC_VERSION
    assert "Authorization" not in headers


def test_the_messages_api_never_receives_a_seed(foundry):
    """It has no such parameter and rejects unknown fields, so sending one
    would fail every call rather than merely be ignored."""
    provider = foundry("https://res.services.ai.azure.com/anthropic")
    _, _, body = provider.build_request("hi", max_tokens=900, temperature=0.0, seed=7)

    assert "seed" not in body
    assert body["max_tokens"] == 900  # required by Messages, unlike Chat Completions
    assert body["temperature"] == 0.0
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_a_chat_completions_endpoint_still_gets_bearer_and_seed(foundry):
    provider = foundry("https://res.services.ai.azure.com/models")
    url, headers, body = provider.build_request(
        "hi", max_tokens=900, temperature=0.0, seed=7
    )

    assert url.endswith("/models/chat/completions")
    assert headers["Authorization"] == "Bearer test-key"
    assert "x-api-key" not in headers
    assert body["seed"] == 7


def test_the_model_id_is_sent_exactly_as_configured(foundry):
    # A routing-prefixed name returns DeploymentNotFound, so the adapter must
    # not quietly rewrite it: the operator has to see their own value echoed
    # back in the error.
    provider = foundry("https://res.services.ai.azure.com/anthropic",
                       model="azure_ai/claude-sonnet-4-6")
    _, _, body = provider.build_request("hi", max_tokens=16, temperature=0.0, seed=None)
    assert body["model"] == "azure_ai/claude-sonnet-4-6"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"content": [{"type": "text", "text": "OK"}]}, "OK"),
        # Several text blocks are concatenated in order.
        ({"content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}, "AB"),
        # A non-text block (a tool call) contributes nothing.
        ({"content": [{"type": "tool_use", "id": "x"}]}, ""),
        ({"choices": [{"message": {"content": "OK"}}]}, "OK"),
        ({"choices": [{"message": {}}]}, ""),
        ({}, ""),
    ],
)
def test_both_response_shapes_are_understood(payload, expected):
    assert AzureAIFoundryLLMProvider.extract_text(payload) == expected


def test_the_adapter_needs_no_sdk():
    """The runtime image installs neither the openai nor the anthropic SDK, so
    an import of either here would fail in exactly the container that needs to
    make the call."""
    import inspect

    source = inspect.getsource(AzureAIFoundryLLMProvider)
    assert "import openai" not in source
    assert "import anthropic" not in source
    assert "urllib" in source


# ─── the suite must never reach a paid API ───────────────────────────────────


def test_the_session_guard_pins_the_provider_to_the_stub():
    """Regression for a real incident: with credentials in .env.local, a bare
    `ReviewAgent()` in a test made a live billable call. tests/conftest.py now
    pins the provider for the whole session."""
    from maxxflow_core.settings import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.llm_provider == "stub"
    assert settings.azure_ai_api_key.get_secret_value() == ""
    assert settings.azure_ai_api_base == ""


def test_a_provider_built_from_configuration_is_the_stub():
    assert isinstance(get_llm_provider(), StubLLMProvider)
