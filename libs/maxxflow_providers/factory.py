"""Provider factories — adapter chosen by config (dict lookup, no env branch)."""

from __future__ import annotations

from maxxflow_core.ports import EmbeddingProvider, LLMProvider
from maxxflow_core.settings import get_settings
from maxxflow_providers.embeddings import AzureOpenAIEmbeddingProvider, StubEmbeddingProvider
from maxxflow_providers.llm import (
    AzureAIFoundryLLMProvider,
    AzureOpenAILLMProvider,
    StubLLMProvider,
)

_LLM = {
    "stub": StubLLMProvider,
    "azure_openai": AzureOpenAILLMProvider,
    "openai": AzureOpenAILLMProvider,  # same OpenAI-compatible client in Phase 2
    # Azure AI Foundry — a separate data plane with its own credentials, and
    # the only one of these that can serve the non-OpenAI catalogue.
    "azure_ai": AzureAIFoundryLLMProvider,
}
_EMB = {
    "stub": StubEmbeddingProvider,
    "azure_openai": AzureOpenAIEmbeddingProvider,
    "openai": AzureOpenAIEmbeddingProvider,
}


def get_llm_provider() -> LLMProvider:
    s = get_settings()
    return _LLM[s.llm_provider]()


def get_embedding_provider() -> EmbeddingProvider:
    s = get_settings()
    cls = _EMB[s.embedding_provider]
    return cls(enabled=s.embeddings_enabled)
