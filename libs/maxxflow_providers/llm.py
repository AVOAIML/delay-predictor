"""LLM providers. Text generation ONLY — never scoring (plan §4 M4, §2)."""

from __future__ import annotations

import hashlib


class StubLLMProvider:
    """Deterministic offline stub. Phrases a human-readable fix explanation from
    the structured fix already produced by the rule engine. No network, no PII."""

    name = "stub"

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        # Deterministic, content-addressed so tests are reproducible.
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        first_line = prompt.strip().splitlines()[0] if prompt.strip() else ""
        return f"[stub-llm:{h}] Suggested explanation for: {first_line[:200]}"


class AzureOpenAILLMProvider:  # pragma: no cover - Phase 2 seam
    """Phase-2 adapter. Azure OpenAI / Foundry. Requires a VERIFIED deployed model
    id (plan §13 / §12a) — do not guess. Construction fails loudly until wired."""

    name = "azure_openai"

    def __init__(self):
        raise NotImplementedError(
            "Azure OpenAI LLM adapter is a Phase-2 seam. Set LLM_MODEL_ID to a "
            "VERIFIED deployed model id and provide credentials (Key Vault) first."
        )

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        raise NotImplementedError
