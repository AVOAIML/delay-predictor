"""maxxflow_providers — external LLM / embeddings behind ports (plan §2, §4 M4).

Classical models do ALL scoring on our infra. Language needs (M4 suggested-fix
phrasing; optional M4 semantic boost) are API calls behind ``LLMProvider`` /
``EmbeddingProvider``. Local/CI uses deterministic STUBS — never a paid API. The
Azure OpenAI / OpenAI / Foundry adapters are Phase-2 seams selected by config
(``LLM_PROVIDER`` / ``EMBEDDING_PROVIDER``), with verify-before-building TODOs on
the model id/region.
"""

from maxxflow_providers.factory import get_embedding_provider, get_llm_provider

__all__ = ["get_llm_provider", "get_embedding_provider"]
